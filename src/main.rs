use clap::Parser;

use rust_htslib::bam::{self, Read, FetchDefinition};
use rust_htslib::bam::pileup::Pileup;
use rust_htslib::bam::pileup::Alignment;
use rust_htslib::faidx;
use std::error::Error;
use std::collections::HashMap;
use std::collections::HashSet;
use statrs::distribution::{Binomial, DiscreteCDF, Discrete};
use std::io::Write;
use rust_htslib::bam::pileup::Indel;
use rayon::prelude::*;
use std::fs::File;
use std::fmt;
use std::hash::{Hash, Hasher};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;


#[derive(Parser, Debug)]
#[command(name = "tvc", about = "A Taps Variant Caller")]
struct Args {
    /// A positional argument
    input_ref: String,
    input_bam: String,
    output_vcf: String,

    #[arg(short = 'b', long, default_value_t = 20)]
    min_bq: usize,

    #[arg(short = 'm', long, default_value_t = 1)]
    min_mapq: usize,

    #[arg(short = 'd', long, default_value_t = 2)]
    min_depth: u32,

    #[arg(short = 'e', long, default_value_t = 5)]
    end_of_read_cutoff: usize,

    #[arg(short = 'i', long, default_value_t = 20)]
    indel_end_of_read_cutoff: usize,

    #[arg(short = 'x', long, default_value_t = 10)]
    max_mismatches: u32,

    #[arg(short = 'a', long, default_value_t = 2)]
    min_ao: u32,

    #[arg(short = 't', long, default_value_t = 4)]
    num_threads: usize,

    #[arg(short = 'c', long, default_value_t = 1000000)]
    chunk_size: u64,

    #[arg(short = 'p', long, default_value_t = 0.005)]
    error_rate: f64,
}

struct Variant {
    contig: String,
    pos: u32,
    reference: String,
    alt: String,
    genotype: String,
    score: f64,
    depth: u32,
    alt_counts: u32,
    calling_directive: CALLING_DIRECTIVE,
}

impl Variant {
    fn new(
        contig: String,
        pos: u32,
        reference: String,
        alt: String,
        genotype: String,
        score: f64,
        depth: u32,
        alt_counts: u32,
        calling_directive: CALLING_DIRECTIVE,

    ) -> Self {
        Variant {
            contig,
            pos,
            reference,
            alt,
            genotype,
            score,
            depth,
            alt_counts,
            calling_directive,
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
            "{}\t{}\t.\t{}\t{}\t{}\t.\tVT={};CD={}\tGT:DP:AO\t{}:{}:{}\n",
            self.contig,
            self.pos,
            self.reference,
            self.alt,
            self.score.round(),
            variant_type,
            match self.calling_directive {
                CALLING_DIRECTIVE::REFERENCE_SITE_OB => "REF_OB",
                CALLING_DIRECTIVE::DENOVO_SITE_OB => "DENOVO_OB",
                CALLING_DIRECTIVE::REFERENCE_SITE_OT => "REF_OT",
                CALLING_DIRECTIVE::DENOVO_SITE_OT => "DENOVO_OT",
                CALLING_DIRECTIVE::BOTH_STRANDS => "BOTH",
            },
            self.genotype,
            self.depth,
            self.alt_counts,
        )
    }
}



struct Genotype{
    genotype: String,
    score: f64 
}

impl Genotype{
    fn new(genotype: &str, best_prob: f64, all_probs_sum: f64) -> Self {
        // normalized probability of the best genotype
        let p_best = best_prob / all_probs_sum;

        // Avoid log10(0) by capping the minimum probability
        let epsilon = 1e-300;
        let p_safe = 1.0 - p_best;
        let p_safe = p_safe.max(epsilon);

        // Phred-scale quality
        let score = -10.0 * p_safe.log10();

        // Cap phred at something reasonable for VCF
        let score = score.min(999.0);
        Genotype { genotype: genotype.to_string(), score }
    }
}

#[derive(Clone)]
enum CALLING_DIRECTIVE {
    REFERENCE_SITE_OB,
    DENOVO_SITE_OB,
    REFERENCE_SITE_OT,
    DENOVO_SITE_OT,
    BOTH_STRANDS,
}




#[derive(Clone)]
struct BaseCall {
    base: char,
    ref_base: char,
    deleted_bases: Vec<u8>,
    insertion_bases: Vec<u8>,
}

impl BaseCall {
    fn new(alignment: &Alignment, ref_seq: &Vec<u8>, ref_pos: u32) -> Self {
        let qpos = alignment.qpos().unwrap();
        let base = alignment.record().seq().as_bytes()[qpos] as char;
        let ref_base = ref_seq[ref_pos as usize] as char;

        let mut deleted_bases = Vec::new();
        let mut insertion_bases = Vec::new();

        match alignment.indel() {
            Indel::Del(len) => {
                let start = ref_pos as usize + 1;
                let end = start + len as usize;
                deleted_bases = ref_seq.get(start..end).unwrap_or(&[]).to_vec();
            }
            Indel::Ins(len) => {
                let read_seq = alignment.record().seq().as_bytes();
                let start = qpos + 1;
                let end = start + len as usize;
                insertion_bases = read_seq.get(start..end).unwrap_or(&[]).to_vec();
            }
            Indel::None => {}
        }

        BaseCall {
            base,
            ref_base,
            deleted_bases,
            insertion_bases,
        }
    }

    fn get_reference_allele(&self) -> String {
        let mut ref_allele = String::new();
        ref_allele.push(self.ref_base);
        if !self.deleted_bases.is_empty() {
            ref_allele.push_str(&String::from_utf8_lossy(&self.deleted_bases));
        }
        ref_allele
    }

    fn get_alternate_allele(&self) -> String {
        let mut alt_allele = String::new();
        alt_allele.push(self.base);
        if !self.insertion_bases.is_empty() {
            alt_allele.push_str(&String::from_utf8_lossy(&self.insertion_bases));
        }
        alt_allele
    }

    fn is_snp(&self) -> bool {
        self.deleted_bases.is_empty() && self.insertion_bases.is_empty()
    }

}

impl fmt::Display for BaseCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Base: {}\tDeleted: {}\tInserted: {}",
            self.base,
            String::from_utf8_lossy(&self.deleted_bases),
            String::from_utf8_lossy(&self.insertion_bases)
        )
    }
}

impl PartialEq for BaseCall {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.deleted_bases == other.deleted_bases
            && self.insertion_bases == other.insertion_bases
    }
}

impl Eq for BaseCall {}

impl Hash for BaseCall {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
        self.deleted_bases.hash(state);
        self.insertion_bases.hash(state);
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



fn find_where_to_call_variants(ref_base: char, alt_candidates: &HashSet<BaseCall>, upstream_base: char, downstream_base: char) -> CALLING_DIRECTIVE {
    
    let alt_candidate_bases: HashSet<char> = alt_candidates.iter().map(|bc| bc.base).collect();
    
    if ref_base == 'C' && downstream_base == 'G'{
        return  CALLING_DIRECTIVE::REFERENCE_SITE_OB;
    }
    else if alt_candidate_bases.contains(&'C') && downstream_base == 'G'{
        return  CALLING_DIRECTIVE::DENOVO_SITE_OB;
    }
    else if ref_base == 'G' && upstream_base == 'C'{
        return  CALLING_DIRECTIVE::REFERENCE_SITE_OT;
    }
    else if alt_candidate_bases.contains(&'G') && upstream_base == 'C'{
        return  CALLING_DIRECTIVE::DENOVO_SITE_OT;
    }
    else {
        return CALLING_DIRECTIVE::BOTH_STRANDS;
    }
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
        ##INFO=<ID=CD,Number=0,Type=String,Description=\"TVC Call Directive\">\n\
        ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n\
        ##FORMAT=<ID=RO,Number=1,Type=Integer,Description=\"Reference Allele Count\">\n\
        ##FORMAT=<ID=AO,Number=1,Type=Integer,Description=\"Alternate Allele Count\">\n\
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
    , contigs)
}

fn right_tail_binomial_pval(n: u64, k: u64, p: f64) -> f64 {
    let binom = Binomial::new(p, n).expect("Failed to create binomial dist");
    let cdf = binom.cdf((k - 1) as u64); 
    1.0 - cdf
}

fn get_count_vec_candidates(counts: &HashMap<BaseCall, usize>, ref_base: char, error_rate: f64) -> HashSet<BaseCall> {
    let mut candidates = HashSet::new();

    let total_depth = counts.values().sum::<usize>() as u64;
    for (basecall, &count) in counts.iter() {

        if (basecall.base == ref_base &&  basecall.is_snp())|| basecall.base == 'N' {
            continue;
        }
        let pval = right_tail_binomial_pval(total_depth, count as u64, error_rate);
        if pval < 0.05 {
            candidates.insert(basecall.clone());
        }
    
    }
    candidates
}

fn assign_genotype(alt_counts: usize, depth: usize, error_rate: f64) -> Genotype {
    let homo_ref_prob = Binomial::new(error_rate, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);
    let het_prob = Binomial::new(0.5, depth as u64).unwrap().pmf(alt_counts as u64);
    let homo_alt_prob = Binomial::new(1.0 - error_rate, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);

    let total = homo_ref_prob + het_prob + homo_alt_prob;

    let (gt, best_prob) = if homo_ref_prob > het_prob && homo_ref_prob > homo_alt_prob {
        ("0/0", homo_ref_prob)
    } else if het_prob > homo_ref_prob && het_prob > homo_alt_prob {
        ("0/1", het_prob)
    } else {
        ("1/1", homo_alt_prob)
    };

    Genotype::new(gt, best_prob, total)
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






fn extract_pileup_counts(pileup: &Pileup, min_bq: usize, min_mapq: usize, end_of_read_cutoff: usize, indel_end_of_read_cutoff: usize, max_mismatches: u32, ref_seq: &Vec<u8>, ref_pos: u32) -> (HashMap<BaseCall, usize>, HashMap<BaseCall, usize>, HashMap<BaseCall, usize>) {
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

            let base_call = BaseCall::new(&alignment, ref_seq, ref_pos);

            let read_len = record.seq().len();
            if base_call.is_snp() {
                
                if qpos < end_of_read_cutoff || qpos >= read_len - end_of_read_cutoff {
                    continue;
                }
            } else {
                if qpos < indel_end_of_read_cutoff || qpos >= read_len - indel_end_of_read_cutoff {
                    continue;
                }
            }
        
            
            if record.is_reverse() && record.is_first_in_template() {
                r_one_r_counts.insert(base_call.clone(), r_one_r_counts.get(&base_call).unwrap_or(&0) + 1);
            } else if !record.is_reverse() && record.is_first_in_template() {
                r_one_f_counts.insert(base_call.clone(), r_one_f_counts.get(&base_call).unwrap_or(&0) + 1);
            } else if record.is_reverse() && !record.is_first_in_template() {
                r_one_f_counts.insert(base_call.clone(), r_one_f_counts.get(&base_call).unwrap_or(&0) + 1);
            } else if !record.is_reverse() && !record.is_first_in_template() {
                r_one_r_counts.insert(base_call.clone(), r_one_r_counts.get(&base_call).unwrap_or(&0) + 1);
            }
            total_counts.insert(base_call.clone(), total_counts.get(&base_call).unwrap_or(&0) + 1);
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
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    min_ao: u32,
    num_threads: usize,
    chunk_size: u64,
    error_rate: f64,
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

    let pb = ProgressBar::new(chunks.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks")?
        .progress_chars("#>-"));


    let max_open_files = 1000;
    let open_files_counter = Arc::new(AtomicUsize::new(0));


    // Rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?;

    let all_variants: Vec<Variant> = pool.install(|| {
    chunks
        .par_iter()
        .map(|chunk| {
            while open_files_counter.load(Ordering::SeqCst) >= max_open_files {
                thread::sleep(Duration::from_millis(1));
            }

            open_files_counter.fetch_add(1, Ordering::SeqCst);
            
            let res = call_variants(
                chunk,
                bam_path,
                seq_name_to_seq.get(&chunk.contig)
                    .expect("Contig not found in reference"),
                min_bq,
                min_mapq,
                min_depth,
                end_of_read_cutoff,
                indel_end_of_read_cutoff,
                max_mismatches,
                min_ao,
                error_rate
            )
            .unwrap_or_else(|e| {
                Vec::new()
            });
            open_files_counter.fetch_sub(1, Ordering::SeqCst);
            pb.inc(1);
            res
        })
        .flatten()
        .collect()
});

    pb.finish_with_message("Variant calling complete");
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


fn call_variants(chunk: &GenomeChunk, bam_path: &str, ref_seq: &Vec<u8>,  min_bq: usize, min_mapq: usize, min_depth: u32, end_of_read_cutoff: usize, indel_end_of_read_cutoff: usize, max_mismatches: u32, min_ao: u32, error_rate: f64) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
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
        
        let (r_one_f_counts, r_one_r_counts, total_counts) = extract_pileup_counts(&pileup, min_bq, min_mapq, end_of_read_cutoff, indel_end_of_read_cutoff, max_mismatches, ref_seq, pos);
        
        let mut all_found_alts: HashSet<&BaseCall> = HashSet::new();
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



        let r_one_f_candidates = get_count_vec_candidates(&r_one_f_counts, ref_base as char, error_rate);
        let r_one_r_candidates = get_count_vec_candidates(&r_one_r_counts, ref_base as char, error_rate);
        
        let directive = find_where_to_call_variants(
    ref_base as char,
    &r_one_f_candidates,
    upstream_base as char,
    downstream_base as char,
);

let (candidates, counts): (HashSet<BaseCall>, HashMap<BaseCall, usize>) = match directive {
    CALLING_DIRECTIVE::REFERENCE_SITE_OB | CALLING_DIRECTIVE::DENOVO_SITE_OB => (
        r_one_r_candidates.clone(),
        r_one_r_counts.clone(),
    ),
    CALLING_DIRECTIVE::REFERENCE_SITE_OT | CALLING_DIRECTIVE::DENOVO_SITE_OT => (
        r_one_f_candidates.clone(),
        r_one_f_counts.clone(),
    ),
    CALLING_DIRECTIVE::BOTH_STRANDS => (
        r_one_f_candidates
            .intersection(&r_one_r_candidates)
            .cloned()
            .collect(),
        total_counts.clone(),
    ),
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
                let genotype = assign_genotype(*alt_counts, total_depth as usize, error_rate);
                if genotype.genotype == "0/0" {
                    continue;
                }

                let variant = Variant::new(
                    ref_name.to_string(),
                    pos + 1, // Convert to 1-based position
                    candidate.get_reference_allele(),
                    candidate.get_alternate_allele(),
                    genotype.genotype,
                    genotype.score,
                    total_depth as u32,
                    *alt_counts as u32,
                    directive.clone(),

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
    let indel_end_of_read_cutoff = args.indel_end_of_read_cutoff;
    let max_mismatches = args.max_mismatches;
    let min_ao = args.min_ao;
    let num_threads = args.num_threads;
    let chunk_size = args.chunk_size;
    let error_rate = args.error_rate;

    workflow(bam_path, ref_path, vcf_path, min_bq, min_mapq, min_depth, end_of_read_cutoff, indel_end_of_read_cutoff, max_mismatches, min_ao, num_threads, chunk_size, error_rate)?;
    
    Ok(())
}
