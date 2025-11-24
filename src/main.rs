use clap::Parser;
use clap::ValueEnum;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rust_htslib::bam::pileup::Alignment;
use rust_htslib::bam::pileup::Indel;
use rust_htslib::bam::pileup::Pileup;
use rust_htslib::bam::{self, Read};
use rust_htslib::faidx;
use rust_htslib::bam::record::Cigar;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, ValueEnum)]
pub enum ReadNumber {
    R1,
    R2,
}

/// Input arguments for the Taps Variant Caller
#[derive(Parser, Debug)]
#[command(name = "tvc", version, about = "A Taps+ Variant Caller")]
struct Args {
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

    #[arg(short = 'i', long, default_value_t = 1)]
    indel_end_of_read_cutoff: usize,

    #[arg(short = 'x', long, default_value_t = 10)]
    max_mismatches: u32,

    #[arg(short = 'a', long, default_value_t = 2)]
    min_ao: u32,

    #[arg(short = 't', long, default_value_t = 4)]
    num_threads: usize,

    #[arg(short = 'c', long, default_value_t = 1000000)]
    chunk_size: u64,

    #[arg(short = 'p', long, default_value_t = 0.05)]
    error_rate: f64,

    #[arg(short = 'r', long, value_enum, default_value_t = ReadNumber::R1)]
    stranded_read: ReadNumber,
}

/// Representation of a genomic variant
///
/// # Fields
/// * `contig` - Chromosome or contig name
/// * `pos` - 1-based position of the variant
/// * `reference` - Reference allele
/// * `alt` - Alternate allele
/// * `genotype` - Genotype string (e.g., "0/1")
/// * `score` - Phred-scaled quality score
/// * `depth` - Read depth at the variant position
/// * `alt_counts` - Count of reads supporting the alternate allele
/// * `calling_directive` - Calling directive for the variant caller
#[derive(Clone, Debug)]
struct Variant {
    contig: String,
    pos: u32,
    reference: String,
    alt: String,
    genotype: String,
    score: f64,
    depth: u32,
    alt_counts: u32,
    calling_directive: CallingDirective,
}

impl Variant {
    /// Create a new Variant instance
    ///
    /// # Arguments
    /// * `contig` - Chromosome or contig name
    /// * `pos` - 1-based position of the variant
    /// * `reference` - Reference allele
    /// * `alt` - Alternate allele
    /// * `genotype` - Genotype string (e.g., "0/1")
    /// * `score` - Phred-scaled quality score
    /// * `depth` - Read depth at the variant position
    /// * `alt_counts` - Count of reads supporting the alternate allele
    /// * `calling_directive` - Calling directive for the variant caller
    ///
    /// # Returns
    /// A new Variant instance
    fn new(
        contig: String,
        pos: u32,
        reference: String,
        alt: String,
        genotype: String,
        score: f64,
        depth: u32,
        alt_counts: u32,
        calling_directive: CallingDirective,
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

    /// Infer the type of variant based on reference and alternate alleles
    ///
    // # Returns
    /// A string representing the variant type (e.g., "SNP", "INS", "DEL", "MNP", "COMPLEX")
    fn infer_variant_type(&self) -> String {
        if self.reference.len() == 1 && self.alt.len() == 1 {
            "SNP".to_string()
        } else if self.reference.len() > 1
            && self.alt.len() > 1
            && self.reference.len() == self.alt.len()
        {
            "MNP".to_string()
        } else if self.reference.len() > 1 && self.alt.len() == 1 {
            "DEL".to_string()
        } else if self.reference.len() == 1 && self.alt.len() > 1 {
            "INS".to_string()
        } else {
            "COMPLEX".to_string()
        }
    }

    /// Convert the Variant instance to a VCF-formatted string
    ///
    /// # Returns
    /// A string in VCF format representing the variant
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
                CallingDirective::ReferenceSiteOb => "REF_OB",
                CallingDirective::DenovoSiteOb => "DENOVO_OB",
                CallingDirective::ReferenceSiteOt => "REF_OT",
                CallingDirective::DenovoSiteOt => "DENOVO_OT",
                CallingDirective::BothStrands => "BOTH",
            },
            self.genotype,
            self.depth,
            self.alt_counts,
        )
    }
}

/// Representation of a genotype with associated quality score
///
/// # Fields
/// * `genotype` - Genotype string (e.g., "0/1")
/// * `score` - Phred-scaled quality score
struct Genotype {
    genotype: String,
    score: f64,
}

impl Genotype {
    /// Create a new Genotype instance with phred-scaled quality score
    ///
    /// # Arguments
    /// * `genotype` - Genotype string (e.g., "0/1")
    /// * `best_prob` - Probability of the best genotype
    /// * `all_probs_sum` - Sum of probabilities of all genotypes
    ///
    /// # Returns
    /// A new Genotype instance with calculated quality score
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
        Genotype {
            genotype: genotype.to_string(),
            score,
        }
    }
}

#[derive(Clone, Debug)]
/// Calling directives for the Taps Variant Caller
///# Variants
/// * `ReferenceSiteOb` - Call at reference site on original bottom strand
/// * `DenovoSiteOb` - Call at de novo site on original bottom strand
/// * `ReferenceSiteOt` - Call at reference site on original top strand
/// * `DenovoSiteOt` - Call at de novo site on original top strand
/// * `BothStrands` - Call on both strands
enum CallingDirective {
    ReferenceSiteOb,
    DenovoSiteOb,
    ReferenceSiteOt,
    DenovoSiteOt,
    BothStrands,
}

#[derive(Clone)]
/// Representation of a base call from a read alignment
///
/// # Fields
/// * `base` - The base called from the read
/// * `ref_base` - The reference base at the position
/// * `deleted_bases` - Bases deleted in the read
/// * `insertion_bases` - Bases inserted in the read
struct BaseCall {
    base: char,
    ref_base: char,
    deleted_bases: Vec<u8>,
    insertion_bases: Vec<u8>,
}

impl BaseCall {
    /// Create a new BaseCall instance from an alignment
    ///
    /// # Arguments
    /// * `alignment` - The pileup alignment
    /// * `ref_seq` - The reference sequence as a byte vector
    /// * `ref_pos` - The reference position
    ///
    /// # Returns
    /// A new BaseCall instance
    fn new(alignment: &Alignment, ref_seq: &[u8], ref_pos: u32) -> Self {
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

    /// Get the reference allele string
    ///
    /// # Returns
    /// A string representing the reference allele
    fn get_reference_allele(&self) -> String {
        let mut ref_allele = String::new();
        ref_allele.push(self.ref_base);
        if !self.deleted_bases.is_empty() {
            ref_allele.push_str(&String::from_utf8_lossy(&self.deleted_bases));
        }
        ref_allele
    }

    /// Get the alternate allele string
    ///
    /// # Returns
    /// A string representing the alternate allele
    fn get_alternate_allele(&self) -> String {
        let mut alt_allele = String::new();
        alt_allele.push(self.base);
        if !self.insertion_bases.is_empty() {
            alt_allele.push_str(&String::from_utf8_lossy(&self.insertion_bases));
        }
        alt_allele
    }

    /// Check if the BaseCall represents a SNP
    /// # Returns
    /// True if SNP, false otherwise
    fn is_snp(&self) -> bool {
        self.deleted_bases.is_empty() && self.insertion_bases.is_empty()
    }
}

impl fmt::Display for BaseCall {
    /// Format the BaseCall for display
    ///
    /// # Returns
    /// A formatted string representation of the BaseCall
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
    /// Compare two BaseCall instances for equality
    ///
    /// # Returns
    /// True if equal, false otherwise
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.deleted_bases == other.deleted_bases
            && self.insertion_bases == other.insertion_bases
    }
}

impl Eq for BaseCall {}

impl Hash for BaseCall {
    /// Hash the BaseCall instance
    ///
    /// # Returns
    /// A hash value for the BaseCall
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
        self.deleted_bases.hash(state);
        self.insertion_bases.hash(state);
    }
}

/// A chunk of the genome for processing
///
/// # Fields
/// * `contig` - Chromosome or contig name
/// * `start` - Start position of the chunk (0-based)
/// * `end` - End position of the chunk (0-based, exclusive)
struct GenomeChunk {
    contig: String,
    start: u64,
    end: u64,
}

impl GenomeChunk {
    /// Create a new GenomeChunk instance
    ///
    /// # Arguments
    /// * `contig` - Chromosome or contig name
    /// * `start` - Start position of the chunk (0-based)
    /// * `end` - End position of the chunk (0-based, exclusive)
    /// # Returns
    /// A new GenomeChunk instance
    fn new(contig: String, start: u64, end: u64) -> Self {
        GenomeChunk { contig, start, end }
    }
}

/// Divide the genome into chunks for processing
///
/// # Arguments
/// * `fasta_path` - Path to the reference FASTA file
/// * `chunk_size` - Size of each chunk
///
/// # Returns
/// A vector of GenomeChunk instances
fn get_genome_chunks(fasta_path: &str, chunk_size: u64) -> Vec<GenomeChunk> {
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

/// Validate that the FAI and BAM headers have matching contigs and lengths
///
/// # Arguments
/// * `fasta_path` - Path to the reference FASTA file
/// * `bam_path` - Path to the BAM file
///
/// # Returns
/// Ok(()) if validation passes, error otherwise
fn validate_fai_and_bam(
    fasta_path: &str,
    bam_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let fai_reader = faidx::Reader::from_path(fasta_path)?;
    let bam_reader = bam::Reader::from_path(bam_path)?;
    let fai_contigs: HashMap<String, u64> = fai_reader
        .seq_names()?
        .iter()
        .map(|name| {
            let len = fai_reader.fetch_seq_len(name);
            (name.clone(), len)
        })
        .collect();
    let bam_header = bam_reader.header();
    for tid in 0..bam_header.target_count() {
        let name = std::str::from_utf8(bam_header.tid2name(tid))?.to_string();
        let len = bam_header.target_len(tid).unwrap();
        match fai_contigs.get(&name) {
            Some(&fai_len) => {
                if fai_len != len {
                    return Err(format!(
                        "Length mismatch for contig {}: FAI length = {}, BAM length = {}",
                        name, fai_len, len
                    )
                    .into());
                }
            }
            None => {
                return Err(format!("Contig {} found in BAM header but not in FAI", name).into());
            }
        }
    }
    Ok(())
}

/// Determine the calling directive based on reference and alternate bases
///
/// # Arguments
/// * `ref_base` - Reference base at the position
/// * `alt_candidates` - Set of alternate base candidates
/// * `upstream_base` - Base upstream of the position
/// * `downstream_base` - Base downstream of the position
///
/// # Returns
/// A CallingDirective indicating where to call variants
fn find_where_to_call_variants(
    ref_base: char,
    alt_candidates: &HashSet<BaseCall>,
    upstream_base: char,
    downstream_base: char,
) -> CallingDirective {
    let alt_candidate_bases: HashSet<char> = alt_candidates.iter().map(|bc| bc.base).collect();

    if ref_base == 'C' && downstream_base == 'G' {
        CallingDirective::ReferenceSiteOb
    } else if alt_candidate_bases.contains(&'C') && downstream_base == 'G' {
        CallingDirective::DenovoSiteOb
    } else if ref_base == 'G' && upstream_base == 'C' {
        CallingDirective::ReferenceSiteOt
    } else if alt_candidate_bases.contains(&'G') && upstream_base == 'C' {
        CallingDirective::DenovoSiteOt
    } else {
        CallingDirective::BothStrands
    }
}

/// Generate the VCF header string based on the BAM header
///
/// # Arguments
/// * `header` - The BAM header view
///
/// # Returns
/// A string representing the VCF header
fn get_vcf_header(header: &bam::HeaderView) -> String {
    let contigs = header
        .target_names()
        .iter()
        .map(|name| {
            let name_str = std::str::from_utf8(name).unwrap();
            let length = header.target_len(header.tid(name).unwrap()).unwrap();
            format!("##contig=<ID={},length={}>", name_str, length)
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "##fileformat=VCFv4.3\n\
        {}\n\
##INFO=<ID=VT,Number=1,Type=String,Description=\"Variant Type\">\n\
##INFO=<ID=CD,Number=1,Type=String,Description=\"TVC Call Directive\">\n\
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n\
##FORMAT=<ID=RO,Number=1,Type=Integer,Description=\"Reference Allele Count\">\n\
##FORMAT=<ID=AO,Number=1,Type=Integer,Description=\"Alternate Allele Count\">\n\
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n",
        contigs
    )
}

/// Calculate the right-tail p-value for a binomial distribution
///
/// # Arguments
/// * `n` - Number of trials
/// * `k` - Number of successes
/// * `p` - Probability of success on each trial
///
/// # Returns
/// Right-tail p-value
fn right_tail_binomial_pval(n: u64, k: u64, p: f64) -> f64 {
    let binom = Binomial::new(p, n).expect("Failed to create binomial dist");
    let cdf = binom.cdf(k - 1);
    1.0 - cdf
}

/// Identify candidate base calls based on statistical significance
///
/// # Arguments
/// * `counts` - A hashmap of BaseCall to their counts
/// * `ref_base` - Reference base at the position
/// * `error_rate` - Expected general error rate
///
/// # Returns
/// A set of candidate BaseCall instances
fn get_count_vec_candidates(
    counts: &HashMap<BaseCall, usize>,
    ref_base: char,
    error_rate: f64,
) -> HashSet<BaseCall> {
    let mut candidates = HashSet::new();

    let total_depth = counts.values().sum::<usize>() as u64;
    for (basecall, &count) in counts.iter() {
        if (basecall.base == ref_base && basecall.is_snp()) || basecall.base == 'N' {
            continue;
        }
        let pval = right_tail_binomial_pval(total_depth, count as u64, error_rate);
        if pval < 0.05 {
            candidates.insert(basecall.clone());
        }
    }
    candidates
}

/// Assign genotype based on binomial probabilities
///
/// # Arguments
/// * `alt_counts` - Count of reads supporting the alternate allele
/// * `depth` - Total read depth at the position
/// * `error_rate` - Expected general error rate
///
/// # Returns
/// A Genotype instance with assigned genotype and quality score
fn assign_genotype(alt_counts: usize, depth: usize, error_rate: f64) -> Genotype {
    let homo_ref_prob = Binomial::new(error_rate, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);
    let het_prob = Binomial::new(0.4, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);
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

/// Retrieve an NM tag from a record
///
/// # Arguments
/// * `record` - The record to retrieve the Tags value from
///
/// # Returns
/// The value of the NM tag
fn get_nm_tag(record: &bam::Record) -> u32 {
    match record.aux(b"NM") {
        Ok(bam::record::Aux::I8(n)) => n as u32,
        Ok(bam::record::Aux::U8(n)) => n as u32,
        Ok(bam::record::Aux::I16(n)) => n as u32,
        Ok(bam::record::Aux::U16(n)) => n as u32,
        Ok(bam::record::Aux::I32(n)) => n as u32,
        Ok(bam::record::Aux::U32(n)) => n,
        _ => panic!("NM tag missing or invalid"),
    }
}

/// Determine if a record is the stranded read
///
/// # Arguments
/// * `record` - The record to asses
/// * `stranded_read` which read is stranded
///
/// # Returns
/// True if the read is the stranded one
fn is_stranded_read(record: &bam::Record, stranded_read: &ReadNumber) -> bool {
    let read_orientation = match record.is_last_in_template() {
        true => ReadNumber::R2,
        false => ReadNumber::R1,
    };

    read_orientation == *stranded_read
}
// check if there is a homopolymer at the start of the read
fn homopolymer_read_start(sequence: &[u8], homopolymer_cutoff: usize) -> bool {
    let len = sequence.len();
    if len < homopolymer_cutoff {
        return false;
    }
    let first_base = sequence[0];
    for i in 1..homopolymer_cutoff {
        if sequence[i] != first_base {
            return false;
        }
    }
    true
}
// check if there is a homopolymer at the end of the read
fn homopolymer_read_end(sequence: &[u8], homopolymer_cutoff: usize) -> bool {
    if sequence.len() < homopolymer_cutoff {
        return false
    }
    let last_base = sequence[sequence.len() - 1];
    for i in (sequence.len() - homopolymer_cutoff)..(sequence.len()) {
        if sequence[i] != last_base {
            return false;
        }
    }
    true
}
// check if there is a dinuc repeat at the start of the read
fn dinuc_repeat_read_start(sequence: &[u8]) -> bool {
    if sequence.len() < 4 {
        return false;
    }

    let first_two = &sequence[0..2]; 

    sequence[2..4] == *first_two
}
// check if there is a dinuc repeat at the end of the read
fn dinuc_repeat_read_end(sequence: &[u8]) -> bool {
    if sequence.len() < 4 {
        return false;
    }
    let start_index = sequence.len() - 4;
    let end_index = sequence.len() - 2;

    let last_two = &sequence[sequence.len() - 2..];

    sequence[start_index..end_index] == *last_two
}
// Check if a record has soft clipping in its CIGAR string
fn check_soft_clip(record: &bam::Record) -> bool {
    for op in record.cigar().iter() {
        if let Cigar::SoftClip(_) = op {
            return true;
        }
    }
    false
}

/// Extract base call counts from a pileup
///
/// # Arguments
/// * `pileup` - The pileup to extract counts from
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `ref_seq` - The reference sequence as a byte vector
/// * `ref_pos` - The reference position
///
/// # Returns
/// A tuple of three hashmaps: (R1 forward counts, R1 reverse counts, total counts)
fn extract_pileup_counts(
    pileup: &Pileup,
    min_bq: usize,
    min_mapq: usize,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    ref_seq: &[u8],
    ref_pos: u32,
    stranded_read: &ReadNumber,
) -> (
    HashMap<BaseCall, usize>,
    HashMap<BaseCall, usize>,
    HashMap<BaseCall, usize>,
) {
    let mut r_one_f_counts = HashMap::new();
    let mut r_one_r_counts = HashMap::new();

    let mut total_counts = HashMap::new();

    for alignment in pileup.alignments() {
        let record = alignment.record();
        let mismatches = get_nm_tag(&record);
        if mismatches > max_mismatches {
            continue;
        }

        if let Some(qpos) = alignment.qpos() {
            let base = record.seq().as_bytes()[qpos] as char;
            let qual = record.qual()[qpos];
            let mapq = record.mapq();
            let is_del = alignment.is_del();
            let is_refskip = alignment.is_refskip();
            let seq = record.seq().as_bytes();

            if homopolymer_read_start(&seq, 3) || homopolymer_read_end(&seq, 3) {
                continue;
            }

            if dinuc_repeat_read_start(&seq) || dinuc_repeat_read_end(&seq) {
                continue;
            }

            if check_soft_clip(&record) {
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

            if record.is_secondary() || record.is_supplementary() || record.is_duplicate() {
                continue;
            }

            let base_call = BaseCall::new(&alignment, ref_seq, ref_pos);

            let read_len = record.seq().len();
            if base_call.is_snp() {
                if qpos < end_of_read_cutoff || qpos >= read_len - end_of_read_cutoff {
                    continue;
                }
            } else if qpos < indel_end_of_read_cutoff || qpos >= read_len - indel_end_of_read_cutoff
            {
                continue;
            }

            let is_stranded_read_status = is_stranded_read(&record, stranded_read);

            if (record.is_reverse() && is_stranded_read_status)
                || (!record.is_reverse() && !is_stranded_read_status)
            {
                r_one_r_counts.insert(
                    base_call.clone(),
                    r_one_r_counts.get(&base_call).unwrap_or(&0) + 1,
                );
            } else {
                r_one_f_counts.insert(
                    base_call.clone(),
                    r_one_f_counts.get(&base_call).unwrap_or(&0) + 1,
                );
            }
            total_counts.insert(
                base_call.clone(),
                total_counts.get(&base_call).unwrap_or(&0) + 1,
            );
        }
    }

    (r_one_f_counts, r_one_r_counts, total_counts)
}

/// Main workflow for variant calling
///
/// # Arguments
/// * `bam_path` - Path to the BAM file
/// * `ref_path` - Path to the reference FASTA file
/// * `vcf_path` - Path to the output VCF file
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `min_depth` - Minimum read depth
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `min_ao` - Minimum alternate allele observations
/// * `num_threads` - Number of threads to use
/// * `chunk_size` - Size of each genome chunk
/// * `error_rate` - Expected general error rate
///
/// # Returns
/// Ok(()) if workflow completes successfully, error otherwise
pub fn workflow(
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
    stranded_read: &ReadNumber,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_fai_and_bam(ref_path, bam_path)?;

    let ref_reader = faidx::Reader::from_path(ref_path)?;
    let contigs: Vec<String> = ref_reader.seq_names()?;

    let mut seq_name_to_seq = HashMap::<String, Vec<u8>>::new();

    for contig in &contigs {
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)?
            .into_iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();
        seq_name_to_seq.insert(contig.clone(), ref_seq);
    }

    let chunks: Vec<GenomeChunk> = get_genome_chunks(ref_path, chunk_size);

    let pb = ProgressBar::new(chunks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks",
            )?
            .progress_chars("#>-"),
    );

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
                    seq_name_to_seq
                        .get(&chunk.contig)
                        .expect("Contig not found in reference"),
                    min_bq,
                    min_mapq,
                    min_depth,
                    end_of_read_cutoff,
                    indel_end_of_read_cutoff,
                    max_mismatches,
                    min_ao,
                    error_rate,
                    stranded_read,
                )
                .unwrap_or_else(|_e| Vec::new());
                open_files_counter.fetch_sub(1, Ordering::SeqCst);
                pb.inc(1);
                res
            })
            .flatten()
            .collect()
    });

    pb.finish_with_message("Variant calling complete. Wrapping up.");

    // Sort all variants by contig and position
    let mut sorted_variants = all_variants;
    sorted_variants.sort_by(|a, b| match a.contig.cmp(&b.contig) {
        std::cmp::Ordering::Equal => a.pos.cmp(&b.pos),
        other => other,
    });

    // Write to VCF
    let mut vcf_file = File::create(vcf_path)?;
    let header = bam::Reader::from_path(bam_path)?.header().to_owned();
    vcf_file.write_all(get_vcf_header(&header).as_bytes())?;

    for variant in sorted_variants {
        vcf_file.write_all(variant.to_vcf().as_bytes())?;
    }

    Ok(())
}

/// Call variants in a given genome chunk
///
/// # Arguments
/// * `chunk` - The genome chunk to process
/// * `bam_path` - Path to the BAM file
/// * `ref_seq` - The reference sequence as a byte vector
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `min_depth` - Minimum read depth
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `min_ao` - Minimum alternate allele observations
/// * `error_rate` - Expected general error rate
///
/// # Returns
/// A vector of Variant instances
fn call_variants(
    chunk: &GenomeChunk,
    bam_path: &str,
    ref_seq: &[u8],
    min_bq: usize,
    min_mapq: usize,
    min_depth: u32,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    min_ao: u32,
    error_rate: f64,
    stranded_read: &ReadNumber,
) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    // Placeholder for the workflow function
    // This is where the main logic of your variant caller would go

    let mut bam = bam::IndexedReader::from_path(bam_path).expect("Error opening BAM file");

    let header = bam.header().to_owned();
    let tid = header
        .tid(chunk.contig.as_bytes())
        .ok_or("Contig not found in BAM header")?;

    // This works if using rust-htslib ≥0.44
    bam.fetch((tid, chunk.start as i64, chunk.end as i64))?;

    let mut variants = Vec::new();

    for result in bam.pileup() {
        let pileup: Pileup = result.expect("Failed to read pileup");
        let tid = pileup.tid();
        let ref_name = std::str::from_utf8(header.tid2name(tid))?;

        let pos = pileup.pos(); // 0-based
        let ref_base = ref_seq[pos as usize];

        let depth = pileup.depth();
        if depth < min_depth {
            continue;
        }

        let (r_one_f_counts, r_one_r_counts, total_counts) = extract_pileup_counts(
            &pileup,
            min_bq,
            min_mapq,
            end_of_read_cutoff,
            indel_end_of_read_cutoff,
            max_mismatches,
            ref_seq,
            pos,
            stranded_read,
        );

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

        let r_one_f_candidates =
            get_count_vec_candidates(&r_one_f_counts, ref_base as char, error_rate);
        let r_one_r_candidates =
            get_count_vec_candidates(&r_one_r_counts, ref_base as char, error_rate);

        let directive = find_where_to_call_variants(
            ref_base as char,
            &r_one_f_candidates,
            upstream_base as char,
            downstream_base as char,
        );

        let (candidates, counts): (HashSet<BaseCall>, HashMap<BaseCall, usize>) = match directive {
            CallingDirective::ReferenceSiteOb | CallingDirective::DenovoSiteOb => {
                (r_one_r_candidates.clone(), r_one_r_counts.clone())
            }
            CallingDirective::ReferenceSiteOt | CallingDirective::DenovoSiteOt => {
                (r_one_f_candidates.clone(), r_one_f_counts.clone())
            }
            CallingDirective::BothStrands => (
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let stranded_read = &args.stranded_read;

    workflow(
        bam_path,
        ref_path,
        vcf_path,
        min_bq,
        min_mapq,
        min_depth,
        end_of_read_cutoff,
        indel_end_of_read_cutoff,
        max_mismatches,
        min_ao,
        num_threads,
        chunk_size,
        error_rate,
        stranded_read,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    /// Unit tests for the variant caller
    use super::*;
    use rust_htslib::faidx;

    macro_rules! make_variant_test {
        // Macro to create variant calling tests
        //
        // # Arguments
        // * `$fn_name` - Name of the test function
        // * `$bam_file` - BAM file to use for the test
        // * `$pos` - Position of the variant
        // * `$ref_base` - Expected reference base
        // * `$alt_base` - Expected alternate base
        // * `$gt` - Expected genotype
        // * `$stranded_read` - Which read is stranded
        //
        // # Returns
        // A test function
        ($fn_name:ident, $bam_file:expr, $pos:expr, $ref_base:expr, $alt_base:expr, $gt:expr, $stranded_read:expr) => {
            #[test]
            fn $fn_name() {
                let test_ref = "test_assets/chr11.fasta";
                let test_bam = concat!("test_assets/testing_bams/", $bam_file);

                // Load reference
                let ref_reader = faidx::Reader::from_path(test_ref).expect("Failed to open FASTA");
                let contig = "chr11";
                let seq_len = ref_reader.fetch_seq_len(contig);
                let ref_seq: Vec<u8> = ref_reader
                    .fetch_seq(contig, 0, seq_len as usize)
                    .expect("Failed to fetch seq")
                    .into_iter()
                    .map(|b| b.to_ascii_uppercase())
                    .collect();

                let chunk = GenomeChunk::new(contig.to_string(), $pos, $pos + 1);

                let variants = call_variants(
                    &chunk,
                    test_bam,
                    &ref_seq,
                    20,    // min_bq
                    1,     // min_mapq
                    1,     // min_depth
                    5,     // end_of_read_cutoff
                    20,    // indel_end_of_read_cutoff
                    10,    // max_mismatches
                    1,     // min_ao
                    0.005, // error_rate
                    &$stranded_read,
                )
                .expect("call_variants failed");

                let matching_variant = variants
                    .iter()
                    .find(|v| v.pos == $pos)
                    .expect("Expected variant not found");

                assert_eq!(matching_variant.contig, contig, "Chromosome mismatch");
                assert_eq!(matching_variant.reference, $ref_base, "REF mismatch");
                assert_eq!(matching_variant.alt, $alt_base, "ALT mismatch");
                assert_eq!(matching_variant.genotype, $gt, "GT mismatch");
            }
        };
    }

    // This test tests a call where methylation is not expected to interfer and is homozygous alt
    make_variant_test!(
        test_both_strands_chr11_8198900_a_c_homo,
        "both_strands_chr11_8198900_A_C_homo.bam",
        8198900,
        "A",
        "C",
        "1/1",
        (ReadNumber::R1)
    );

    // This test tests a call where methylation is not expected to interfer and is heterozygous
    make_variant_test!(
        test_both_strands_chr11_8198951_t_a_het,
        "both_strands_chr11_8198951_T_A_het.bam",
        8198951,
        "T",
        "A",
        "0/1",
        (ReadNumber::R1)
    );

    // This test tests a call where there was a denovo CpG created and is homozygous alt where OB is expected to be non-methylated
    make_variant_test!(
        test_denovo_ob_chr11_134755809_t_c_homo,
        "denovo_ob_chr11_134755809_T_C_homo.bam",
        134755809,
        "T",
        "C",
        "1/1",
        (ReadNumber::R1)
    );

    // This test tests a call where there was a denovo CpG created and is heterozygous where OB is expected to be non-methylated
    make_variant_test!(
        test_denovo_ob_chr11_134911365_t_c_het,
        "denovo_ob_chr11_134911365_T_C_het.bam",
        134911365,
        "T",
        "C",
        "0/1",
        (ReadNumber::R1)
    );

    // This test tests a call where there was a denovo CpG created and is heterozygous where OT is expected to be non-methylated
    make_variant_test!(
        test_denovo_ot_chr11_134749303_a_g_het,
        "denovo_ot_chr11_134749303_A_G_het.bam",
        134749303,
        "A",
        "G",
        "0/1",
        (ReadNumber::R1)
    );

    // This test tests a call where there was a denovo CpG created and is homozygous alt where OT is expected to be non-methylated
    make_variant_test!(
        test_denovo_ot_chr11_134479860_a_g_homo,
        "denovo_ot_chr11_134479860_A_G_homo.bam",
        134479860,
        "A",
        "G",
        "1/1",
        (ReadNumber::R1)
    );

    // This test tests a reference CpG site where there is a heterozygous snp and OB expected to be the non-methylated strand
    make_variant_test!(
        test_ref_ob_chr11_134012307_c_a_het,
        "ref_ob_chr11_134012307_C_A_het.bam",
        134012307,
        "C",
        "A",
        "0/1",
        (ReadNumber::R1)
    );

    // This test tests a reference CpG site where there is a homozygous alt snp and OB expected to be the non-methylated strand
    make_variant_test!(
        test_ref_ob_chr11_134610622_c_t_homo,
        "ref_ob_chr11_134610622_C_T_homo.bam",
        134610622,
        "C",
        "T",
        "1/1",
        (ReadNumber::R1)
    );

    // This test tests a reference CpG site where there is a homozygous alt snp and OT expected to be the non-methylated strand
    make_variant_test!(
        test_ref_ot_chr11_134473154_g_a_homo,
        "ref_ot_chr11_134473154_G_A_homo.bam",
        134473154,
        "G",
        "A",
        "1/1",
        (ReadNumber::R1)
    );

    // This test tests a reference CpG site where there is a heterozygous snp and OT expected to be the non-methylated strand
    make_variant_test!(
        test_ref_ot_chr11_8195526_g_a_het,
        "ref_ot_chr11_8195526_G_A_het.bam",
        8195526,
        "G",
        "A",
        "0/1",
        (ReadNumber::R1)
    );

    #[test]
    fn test_methylation_site_no_variants() {
        // Tests a fully methylated site where the are no variants expected
        let test_ref = "test_assets/chr11.fasta";
        let test_bam = "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.bam";

        // Load reference
        let ref_reader = faidx::Reader::from_path(test_ref).expect("Failed to open FASTA");
        let contig = "chr11";
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .into_iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();

        // Define chunk covering the whole methylation region
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            test_bam,
            &ref_seq,
            20,              // min_bq
            1,               // min_mapq
            1,               // min_depth
            5,               // end_of_read_cutoff
            20,              // indel_end_of_read_cutoff
            10,              // max_mismatches
            1,               // min_ao
            0.005,           // error_rate
            &ReadNumber::R1, // stranded_read
        )
        .expect("call_variants failed");

        let filtered_variants: Vec<&Variant> = variants
            .iter()
            .filter(|v| v.pos >= 134755601 && v.pos <= 134755621)
            .collect();

        assert!(
            filtered_variants.is_empty(),
            "Expected no variants in methylation site BAM"
        );
    }

    #[test]
    fn test_single_ended_reads() {
        // Tests that single-ended reads are handled correctly
        let test_ref = "test_assets/chr11.fasta";
        let test_bam =
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.single_end.bam";
        // Load reference
        let ref_reader = faidx::Reader::from_path(test_ref).expect(
            "Failed to
    open FASTA",
        );
        let contig = "chr11";
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .into_iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();

        // Define chunk covering the whole methylation region
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            test_bam,
            &ref_seq,
            20,              // min_bq
            1,               // min_mapq
            1,               // min_depth
            5,               // end_of_read_cutoff
            20,              // indel_end_of_read_cutoff
            10,              // max_mismatches
            1,               // min_ao
            0.005,           // error_rate
            &ReadNumber::R1, // stranded_read
        )
        .expect("call_variants failed");

        let filtered_variants: Vec<&Variant> = variants
            .iter()
            .filter(|v| v.pos >= 134755601 && v.pos <= 134755621)
            .collect();

        assert!(
            filtered_variants.is_empty(),
            "Expected no variants in single-ended methylation site BAM"
        );
    }

    #[test]
    fn test_read_two_stranded() {
        // Tests that Read 2 stranded reads are handled correctly
        let test_ref = "test_assets/chr11.fasta";
        let test_bam = "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.bam";
        // Load reference
        let ref_reader = faidx::Reader::from_path(test_ref).expect(
            "Failed to
    open FASTA",
        );
        let contig = "chr11";
        let seq_len = ref_reader.fetch_seq_len(contig);
        let ref_seq: Vec<u8> = ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .into_iter()
            .map(|b| b.to_ascii_uppercase())
            .collect();

        // Define chunk covering the whole methylation region
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);
        let variants = call_variants(
            &chunk,
            test_bam,
            &ref_seq,
            20,              // min_bq
            1,               // min_mapq
            1,               // min_depth
            5,               // end_of_read_cutoff
            20,              // indel_end_of_read_cutoff
            10,              // max_mismatches
            1,               // min_ao
            0.005,           // error_rate
            &ReadNumber::R2, // stranded_read
        )
        .expect("call_variants failed");
        let filtered_variants: Vec<&Variant> = variants
            .iter()
            .filter(|v| v.pos >= 134755601 && v.pos <= 134755621)
            .collect();

        assert!(
            filtered_variants.len() == 2,
            "Since the R2 was flipped the caller should call these instead got {}",
            filtered_variants.len()
        );
    }

    #[test]
    fn test_homopolymer_read_start() {

        let seq = b"AAATGCC";
        assert!(homopolymer_read_start(seq, 3));

        let seq2 = b"AATGCC";
        assert!(!homopolymer_read_start(seq2, 3));

        let seq3 = b"ATATAT";
        assert!(!homopolymer_read_start(seq3, 3));
    }

    #[test]
    fn test_homopolymer_read_end() {

        let seq = b"GCCTTT";
        assert!(homopolymer_read_end(seq, 3));

        let seq2 = b"GCCTT";
        assert!(!homopolymer_read_end(seq2, 3));

        let seq3 = b"GCCTTA";
        assert!(!homopolymer_read_end(seq3, 3));
    }

    #[test]
    fn test_dinuc_repeat_read_start() {

        let seq = b"ATATGC";
        assert!(dinuc_repeat_read_start(seq));

        let seq2 = b"TGTGAA";
        assert!(dinuc_repeat_read_start(seq2));

        let seq3 = b"ATCGTG";
        assert!(!dinuc_repeat_read_start(seq3));

        let seq4 = b"ATG";
        assert!(!dinuc_repeat_read_start(seq4));
    }

    #[test]
    fn test_dinuc_repeat_read_end() {

        let seq = b"GCCGCG";
        assert!(dinuc_repeat_read_end(seq));

        let seq2 = b"ATTTTT";
        assert!(dinuc_repeat_read_end(seq2));

        let seq3 = b"GGATCC";
        assert!(!dinuc_repeat_read_end(seq3));

        let seq4 = b"ATG";
        assert!(!dinuc_repeat_read_end(seq4));
    }

    #[test]
    fn test_filters_do_not_trigger_falsely() {
        let seq = b"ATGCGT";
        assert!(!homopolymer_read_start(seq, 3));
        assert!(!homopolymer_read_end(seq, 3));
        assert!(!dinuc_repeat_read_start(seq));
        assert!(!dinuc_repeat_read_end(seq));
    }
    
    #[derive(Debug)]
    struct Qualities(Vec<u8>);
    impl Qualities {
        fn from_bytes(bytes: Vec<u8>) -> Self {
            Qualities(bytes)
        }
    }

    #[test]
    fn test_check_soft_clip() {
        let mut record = bam::Record::new();
        let cigar_with_soft_clip = bam::record::CigarString::from(vec![
            Cigar::SoftClip(5),
            Cigar::Match(10),
            Cigar::SoftClip(3),
        ]);
        let qname = b"simulated_read";
        let seq = b"AAAAAAAAAA";
        let quals = Qualities::from_bytes(vec![255; 10]);
        let qual: Vec<u8> = quals.0;
        record.set(qname, Some(&cigar_with_soft_clip), seq, &qual);
        assert!(check_soft_clip(&record));

        let mut record_no_soft_clip = bam::Record::new();
        let cigar_no_soft_clip = bam::record::CigarString::from(vec![
            Cigar::Match(10),
        ]);
        record_no_soft_clip.set(qname, Some(&cigar_no_soft_clip), seq, &qual);
        assert!(!check_soft_clip(&record_no_soft_clip));
    }
}
