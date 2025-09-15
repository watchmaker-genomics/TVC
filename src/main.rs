use clap::Parser;

use rust_htslib::bam::{self, Read, FetchDefinition};
use rust_htslib::bam::pileup::Pileup;
use rust_htslib::faidx;
use std::error::Error;
use std::collections::HashMap;
use std::collections::HashSet;
use statrs::distribution::{Binomial, DiscreteCDF, Discrete};
use std::io::Write;
use rust_htslib::bam::pileup::Indel;
use rayon::prelude::*;
use std::fs::File;


#[derive(Parser, Debug)]
#[command(name = "tvc", about = "A Taps Variant Caller")]
struct Args {
    /// A positional argument
    input_ref: String,
    input_bam: String,
    output_vcf: String,

    #[arg(short = 'b', long, default_value_t = 20)]
    min_bq: usize,

    #[arg(short = 'm', long, default_value_t = 20)]
    min_mapq: usize,

    #[arg(short = 'd', long, default_value_t = 10)]
    min_depth: u32,

    #[arg(short = 'e', long, default_value_t = 20)]
    end_of_read_cutoff: usize,

    #[arg(short = 'x', long, default_value_t = 10)]
    max_mismatches: u32,

    #[arg(short = 'a', long, default_value_t = 2)]
    min_ao: u32,

    #[arg(short = 't', long, default_value_t = 4)]
    num_threads: usize,

    #[arg(short = 'c', long, default_value_t = 1000000)]
    chunk_size: u64,
}

struct Variant {
    contig: String,
    pos: u32,
    reference: String,
    alt: String,
    genotype: String,
    depth: u32,
    alt_counts: u32,
}

impl Variant {
    fn new(
        contig: String,
        pos: u32,
        reference: String,
        alt: String,
        genotype: String,
        depth: u32,
        alt_counts: u32,

    ) -> Self {
        Variant {
            contig,
            pos,
            reference,
            alt,
            genotype,
            depth,
            alt_counts,
        }
    }

    fn infer_variant_type(&self) -> String {
        if self.reference.len() == 1 && self.alt.len() == 1 {
            return "SNP".to_string();
        } else if self.reference.len() > 1 && self.alt.len() > 1  && self.reference.len() == self.alt.len() {
            return "MNP".to_string();
        } else if self.reference.len() > 1 && self.alt.len() == 1 {
            return "DEL".to_string();
        } else if self.reference.len() == 1 && self.alt.len() > 1 {
            return "INS".to_string();
        }
        else {
            return "COMPLEX".to_string();
        }
    }

    fn to_vcf(&self) -> String {

        let variant_type = self.infer_variant_type();

        format!(
            "{}\t{}\t.\t{}\t{}\t100\t.\tVT={}\tGT:DP:AO\t{}:{}:{}\n",
            self.contig,
            self.pos,
            self.reference,
            self.alt,
            variant_type,
            self.genotype,
            self.depth,
            self.alt_counts,
        )
    }
}

struct GenomeChunk {
    contig: String,
    start: u64,
    end: u64,
}

impl GenomeChunk {
    fn new(contig: String, start: u64, end: u64) -> Self {
        GenomeChunk { contig, start, end }
    }
}

fn get_genome_chunks(
    fasta_path: &str,
    chunk_size: u64,
) -> Vec<GenomeChunk> {
    let reader = faidx::Reader::from_path(fasta_path).expect("Failed to open FASTA file");
    let seq_names = reader.seq_names().expect("Failed to get sequence names");

    let mut chunks = Vec::new();
    for seq_name in seq_names {
        let seq_len = reader.fetch_seq_len(&seq_name);
        let mut start = 0;
        while start < seq_len {
            let end = (start + chunk_size).min(seq_len);
            chunks.push(GenomeChunk::new(seq_name.clone(), start, end));
            start += chunk_size;
        }
    }
    chunks
}



fn find_where_to_call_variants(ref_base: char, alt_candidates: &HashSet<char>, upstream_base: char, downstream_base: char) -> String {
    if (ref_base == 'C'|| alt_candidates.contains(&'C')) && downstream_base == 'G'{
        return  "OB".to_string();
    }
    else if (ref_base == 'G'|| alt_candidates.contains(&'G')) && upstream_base == 'C'{
        return  "OT".to_string();
    }
    return "BOTH".to_string();
}

fn get_vcf_header(header: &bam::HeaderView) -> String {
    

    let contigs = header.target_names().iter()
        .map(|name| {
            let name_str = std::str::from_utf8(name).unwrap();
            let length = header.target_len(header.tid(name).unwrap()).unwrap();
            format!("##contig=<ID={},length={}>", name_str, length)
        })
        .collect::<Vec<_>>()
        .join("\n");



    format!(
        "##fileformat=VCFv4.2\n\
        {}
##INFO=<ID=VT,Number=1,Type=String,Description=\"Variant Type\">\n\
        ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n\
        ##FORMAT=<ID=RO,Number=1,Type=Integer,Description=\"Reference Allele Count\">\n\
        ##FORMAT=<ID=AO,Number=1,Type=Integer,Description=\"Alternate Allele Count\">\n\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
    , contigs)
}

fn right_tail_binomial_pval(n: u64, k: u64, p: f64) -> f64 {
    let binom = Binomial::new(p, n).expect("Failed to create binomial dist");
    let cdf = binom.cdf((k - 1) as u64); // P(X < k)
    1.0 - cdf                          // P(X ≥ k)
}

fn get_count_vec_candidates(counts: &HashMap<char, usize>, ref_base: char) -> HashSet<char> {
    let mut candidates = HashSet::new();

    let total_depth = counts.values().sum::<usize>() as u64;
    let p = 0.005; // Placeholder for the probability of success
    for (&base, &count) in counts.iter() {

        if base == ref_base || base == 'N' {
            continue;
        }
        let pval = right_tail_binomial_pval(total_depth, count as u64, p);
        if pval < 0.05 {
            candidates.insert(base);
        }
    
    }
    candidates
}

fn assign_genotype(
    alt_counts: usize,
    depth: usize,
) -> &'static str {
    let p = 0.005; // Placeholder for the probability of success

    let homo_ref_binom = Binomial::new(p, depth.try_into().unwrap()).expect("Failed to create binomial dist");
    let homo_ref_prob = homo_ref_binom.pmf(alt_counts.try_into().unwrap());
    
    let het_binom = Binomial::new(0.5, depth.try_into().unwrap()).expect("Failed to create binomial dist");
    let het_binom_prob = het_binom.pmf(alt_counts.try_into().unwrap());
    
    let alt_prob = 1.0 - p as f64;
    let homo_alt_binom = Binomial::new(alt_prob, depth.try_into().unwrap()).expect("Failed to create binomial dist");
    let homo_alt_prob = homo_alt_binom.pmf(alt_counts.try_into().unwrap());

    if homo_ref_prob > het_binom_prob && homo_ref_prob > homo_alt_prob {
        return "0/0"

    } else if het_binom_prob > homo_ref_prob && het_binom_prob > homo_alt_prob {
        return "0/1"

    } else {
        return "1/1"

    }

}

fn get_nm_tag(record: &bam::Record) -> u32 {
    match record.aux(b"NM") {
        Ok(bam::record::Aux::I8(n)) => n as u32,
        Ok(bam::record::Aux::U8(n)) => n as u32,
        Ok(bam::record::Aux::U16(n)) => n as u32,
        Ok(bam::record::Aux::U32(n)) => n,
        _ => panic!("NM tag missing or invalid"),
    }
}


fn extract_pileup_counts(pileup: &Pileup, min_bq: usize, min_mapq: usize, end_of_read_cutoff: usize, max_mismatches: u32) -> (HashMap<char, usize>, HashMap<char, usize>, HashMap<char, usize>) {
    let mut r_one_f_counts = HashMap::new();
    let mut r_one_r_counts = HashMap::new();
    let mut total_counts = HashMap::new();
    

    for alignment in pileup.alignments() {

        let record = alignment.record();
        let mismatches = get_nm_tag(&record);
        if mismatches > max_mismatches{
            continue;
        }

        if let Some(qpos) = alignment.qpos() {

            let base = record.seq().as_bytes()[qpos] as char;
            let qual = record.qual()[qpos];
            let mapq = record.mapq();
            let is_del = alignment.is_del();
            let is_refskip = alignment.is_refskip();

            if qpos < end_of_read_cutoff{
                continue;
            }

            if record.seq().len() - end_of_read_cutoff < qpos {
                continue;
            }

            if is_del || is_refskip {
                continue;
            }
            if base == 'N' {
                continue;
            }

            if qual < min_bq as u8 {
                continue;
            }
            if mapq < min_mapq as u8 {
                continue;
            }

            if record.is_secondary() || record.is_supplementary() {
                continue;
            }

            

            if record.is_reverse() && record.is_first_in_template() {
                r_one_r_counts.insert(base, r_one_r_counts.get(&base).unwrap_or(&0) + 1);
            } else if !record.is_reverse() && record.is_first_in_template() {
                r_one_f_counts.insert(base, r_one_f_counts.get(&base).unwrap_or(&0) + 1);
            } else if record.is_reverse() && !record.is_first_in_template() {
                r_one_f_counts.insert(base, r_one_f_counts.get(&base).unwrap_or(&0) + 1);
            } else if !record.is_reverse() && !record.is_first_in_template() {
                r_one_r_counts.insert(base, r_one_r_counts.get(&base).unwrap_or(&0) + 1);
            }
            total_counts.insert(base, total_counts.get(&base).unwrap_or(&0) + 1);
        }
    }

    (r_one_f_counts, r_one_r_counts, total_counts)
}

fn workflow(
    bam_path: &str,
    ref_path: &str,
    vcf_path: &str,
    min_bq: usize,
    min_mapq: usize,
    min_depth: u32,
    end_of_read_cutoff: usize,
    max_mismatches: u32,
    min_ao: u32,
    num_threads: usize,
    chunk_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_reader = faidx::Reader::from_path(ref_path)?;
    let contigs: Vec<String> = ref_reader.seq_names()?;

    let mut seq_name_to_seq = HashMap::<String, Vec<u8>>::new();
    
    for contig in &contigs {
        
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader.fetch_seq(contig, 0, seq_len as usize)?.into_iter().map(|b| b.to_ascii_uppercase()).collect();
        seq_name_to_seq.insert(contig.clone(), ref_seq);
    }

    let chunks: Vec<GenomeChunk> = get_genome_chunks(ref_path, chunk_size);



    // Rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?;

    let all_variants: Vec<Variant> = pool.install(|| {
    chunks
        .par_iter()
        .map(|chunk| {
            call_variants(
                chunk,
                bam_path,
                seq_name_to_seq.get(&chunk.contig)
                    .expect("Contig not found in reference"),
                min_bq,
                min_mapq,
                min_depth,
                end_of_read_cutoff,
                max_mismatches,
                min_ao,
            )
            .unwrap_or_else(|e| {
                Vec::new()
            })
        })
        .flatten()
        .collect()
});

    println!("Found {} variants", all_variants.len());


    // Sort all variants by contig and position
    let mut sorted_variants = all_variants;
    sorted_variants.sort_by(|a, b| match a.contig.cmp(&b.contig) {
        std::cmp::Ordering::Equal => a.pos.cmp(&b.pos),
        other => other,
    });

    println!("Sorted variants");

    // Write to VCF
    let mut vcf_file = File::create(vcf_path)?;
    let header = bam::Reader::from_path(bam_path)?.header().to_owned();
    vcf_file.write_all(get_vcf_header(&header).as_bytes())?;

    for variant in sorted_variants {
        vcf_file.write_all(variant.to_vcf().as_bytes())?;
    }

    Ok(())
}


fn call_variants(chunk: &GenomeChunk, bam_path: &str, ref_seq: &Vec<u8>,  min_bq: usize, min_mapq: usize, min_depth: u32, end_of_read_cutoff: usize, max_mismatches: u32, min_ao: u32) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    // Placeholder for the workflow function
    // This is where the main logic of your variant caller would go
    
    println!("Processing chunk: {}:{}-{}", chunk.contig, chunk.start, chunk.end);
    
    let mut bam = bam::IndexedReader::from_path(bam_path).expect("Error opening BAM file");
    
    let header = bam.header().to_owned();
    let tid = header.tid(chunk.contig.as_bytes())
    .ok_or("Contig not found in BAM header")?;

    // This works if using rust-htslib ≥0.44
    bam.fetch((tid, chunk.start as i64, chunk.end as i64))?;


    let mut variants = Vec::new();
    

    let mut position_counter = 0;
    for result in bam.pileup() {
        let pileup: Pileup = result.expect("Failed to read pileup");
        let tid = pileup.tid();
        let ref_name = std::str::from_utf8(header.tid2name(tid as u32))?;



        let pos = pileup.pos(); // 0-based
        let ref_base = ref_seq[pos as usize];

        
        let depth = pileup.depth();
        if depth < min_depth {
            continue;
        }
        
        let (r_one_f_counts, r_one_r_counts, total_counts) = extract_pileup_counts(&pileup, min_bq, min_mapq, end_of_read_cutoff, max_mismatches);
        
        let mut all_found_alts: HashSet<char> = HashSet::new();
        all_found_alts.extend(r_one_f_counts.keys());
        all_found_alts.extend(r_one_r_counts.keys());

        let upstream_base = if pos > 0 {
            ref_seq[pos as usize - 1]
        } else {
            b'N'
        };
        let downstream_base = if pos < ref_seq.len() as u32 - 1 {
            ref_seq[pos as usize + 1]
        } else {
            b'N'
        };



        let r_one_f_candidates = get_count_vec_candidates(&r_one_f_counts, ref_base as char);
        let r_one_r_candidates = get_count_vec_candidates(&r_one_r_counts, ref_base as char);
        
        let (candidates, counts): (HashSet<char>, HashMap<char, usize>) =
    match find_where_to_call_variants(
        ref_base as char,
        &r_one_f_candidates,
        upstream_base as char,
        downstream_base as char,
    ).as_str() {
        "OB" => (
            r_one_r_candidates.clone(),
            r_one_r_counts.clone(),
        ),
        "OT" => (
            r_one_f_candidates.clone(),
            r_one_f_counts.clone(),
        ),
        "BOTH" => (
            r_one_f_candidates
                .intersection(&r_one_r_candidates)
                .cloned()
                .collect(),
            total_counts.clone(),
        ),
        _ => panic!("Invalid variant calling strategy"),
    };

        let total_depth = counts.values().sum::<usize>() as u64;
        if total_depth < min_depth as u64 {
            continue;
        }
        
        
        
        if !candidates.is_empty() {
            for candidate in candidates {
                let alt_counts = counts.get(&candidate).unwrap_or(&0);
                if *alt_counts < min_ao as usize {
                    continue;
                }
                let genotype = assign_genotype(*alt_counts, total_depth as usize);
                if genotype == "0/0" {
                    continue;
                }

                let variant = Variant::new(
                    ref_name.to_string(),
                    pos + 1, // Convert to 1-based position
                    String::from_utf8_lossy(&[ref_base]).to_string(),
                    candidate.to_string(),
                    genotype.to_string(),
                    total_depth as u32,
                    *alt_counts as u32,
                );
                variants.push(variant);
            }
        }


    }
    Ok(variants)
}


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let args = Args::parse();
    let bam_path = &args.input_bam;
    let vcf_path = &args.output_vcf;
    let min_bq = args.min_bq;
    let min_mapq = args.min_mapq;
    let min_depth = args.min_depth;
    let ref_path = &args.input_ref;
    let end_of_read_cutoff = args.end_of_read_cutoff;
    let max_mismatches = args.max_mismatches;
    let min_ao = args.min_ao;
    let num_threads = args.num_threads;
    let chunk_size = args.chunk_size;

    workflow(bam_path, ref_path, vcf_path, min_bq, min_mapq, min_depth, end_of_read_cutoff, max_mismatches, min_ao, num_threads, chunk_size)?;
    
    Ok(())
}
