use clap::Parser;
use clap::ValueEnum;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rust_htslib::bam::pileup::Alignment;
use rust_htslib::bam::pileup::Indel;
use rust_htslib::bam::pileup::Pileup;
use rust_htslib::bam::record::Cigar;
use rust_htslib::bam::{self, Read};
use rust_htslib::faidx;
use statrs::distribution::{Binomial, Discrete, DiscreteCDF};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;
use tracing::info;

const MODEL_PROBABILITY_THRESHOLD: f64 = 0.3;
const MODEL_TNC_BASES: [char; 5] = ['A', 'C', 'G', 'T', 'N'];
const MODEL_VT_VALUES: [&str; 5] = ["COMPLEX", "DEL", "INS", "MNP", "SNP"];

// This must match the exported ONNX model input column order.
// NUMERIC_FEATURES + engineered terms + one-hot categorical dummies.
const MODEL_FEATURE_ORDER: [&str; 53] = [
    "DP", "AO", "ER", "PR",
    "MFR", "MFA", "BFR", "BFA",
    "AMQR", "AMQA", "ABQR", "ABQA",
    "REDR", "REDA", "ISR", "ISA",
    "FWDP", "REVP", "LLE", "SLE",
    "REFC", "AMPR", "MFC", "ARL",
    "FWD", "REV", "TOT",
    "AF", "MQ_diff", "BQ_diff", "RED_diff", "IS_diff", "strand_bias",
    "TNC_up_A", "TNC_up_C", "TNC_up_G", "TNC_up_T", "TNC_up_N",
    "TNC_ref_A", "TNC_ref_C", "TNC_ref_G", "TNC_ref_T", "TNC_ref_N",
    "TNC_down_A", "TNC_down_C", "TNC_down_G", "TNC_down_T", "TNC_down_N",
    "VT_COMPLEX", "VT_DEL", "VT_INS", "VT_MNP", "VT_SNP",
];

#[derive(Debug, Clone, PartialEq, ValueEnum)]
pub enum ReadNumber {
    R1,
    R2,
}

#[derive(ValueEnum, Clone, Debug)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
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

    #[arg(short = 'i', long, default_value_t = 0)]
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

    #[arg(short = 'f', long, default_value_t = 3)]
    indel_filter_repeat_limit: usize,

    #[arg(short = 'r', long, value_enum, default_value_t = ReadNumber::R1)]
    stranded_read: ReadNumber,

    #[arg(short = 'l', long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,

    #[arg(short = 'k', long, default_value = "model.onnx")]
    model_path: String,

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
    error_rate: f64,
    tnc: TrinucleotideContext,
    probability: f64,
    mapq_filtered_ref: f64,
    mapq_filtered_alt: f64,
    bq_filtered_ref: f64,
    bq_filtered_alt: f64,
    average_ref_mapq: f64,
    average_alt_mapq: f64, 
    average_ref_bq: f64,
    average_alt_bq: f64,
    avg_ref_dist_from_read_end: f64,
    avg_alt_dist_from_read_end: f64,
    avg_ref_insert_size: f64,
    avg_alt_insert_size: f64,
    fwd_probability: f64,
    rev_probability: f64,
    large_local_entropy: f64,
    small_local_entropy: f64,
    read_end_filtered_count: f64,
    avg_mismatch_per_read: f64,
    mismatch_filtered_count: f64,
    avg_read_length: f64,
    forward_strand_count_snps: f64,
    reverse_strand_count_snps: f64,
    both_strands_count_snps: f64,
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
        error_rate: f64,
        tnc: TrinucleotideContext,
        probability: f64,
        mapq_filtered_ref: f64,
        mapq_filtered_alt: f64,
        bq_filtered_ref: f64,
        bq_filtered_alt: f64,
        average_ref_mapq: f64,
        average_alt_mapq: f64,
        average_ref_bq: f64,
        average_alt_bq: f64,
        avg_ref_dist_from_read_end: f64,
        avg_alt_dist_from_read_end: f64,
        avg_ref_insert_size: f64,
        avg_alt_insert_size: f64,
        fwd_probability: f64,
        rev_probability: f64,
        large_local_entropy: f64,
        small_local_entropy: f64,
        read_end_filtered_count: f64,
        avg_mismatch_per_read: f64,
        mismatch_filtered_count: f64,
        avg_read_length: f64,
        forward_strand_count_snps: f64,
        reverse_strand_count_snps: f64,
        both_strands_count_snps: f64,
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
            error_rate,
            tnc,
            probability,
            mapq_filtered_ref,
            mapq_filtered_alt,
            bq_filtered_ref,
            bq_filtered_alt,
            average_ref_mapq,
            average_alt_mapq,
            average_ref_bq,
            average_alt_bq,
            avg_ref_dist_from_read_end,
            avg_alt_dist_from_read_end,
            avg_ref_insert_size,
            avg_alt_insert_size,
            fwd_probability,
            rev_probability,
            large_local_entropy,
            small_local_entropy,
            read_end_filtered_count,
            avg_mismatch_per_read,
            mismatch_filtered_count,
            avg_read_length,
            forward_strand_count_snps,
            reverse_strand_count_snps,
            both_strands_count_snps,
        }
    }

    /// Infer the type of variant based on reference and alternate alleles
    ///
    // # Returns
    /// A string representing the variant type (e.g., "SNP", "INS", "DEL", "MNP", "COMPLEX")
    fn infer_variant_type(&self) -> &'static str {
        let rlen = self.reference.len();
        let alen = self.alt.len();
        match (rlen, alen) {
            (1, 1) => "SNP",
            (r, a) if r > 1 && a > 1 && r == a => "MNP",
            (r, 1) if r > 1 => "DEL",
            (1, a) if a > 1 => "INS",
            _ => "COMPLEX",
        }
    }
    /// Render this variant as a VCF record line (newline-terminated).
    fn to_vcf(&self) -> String {
        let cd = match self.calling_directive {
            CallingDirective::ReferenceSiteOb => "REF_OB",
            CallingDirective::DenovoSiteOb => "DENOVO_OB",
            CallingDirective::ReferenceSiteOt => "REF_OT",
            CallingDirective::DenovoSiteOt => "DENOVO_OT",
            CallingDirective::BothStrands | CallingDirective::Indel => "BOTH",
        };

        // Clamp zero probabilities to a small floor so downstream tools can
        // take log without hitting -inf.
        let prob = self.probability.max(1e-300);
        let fwd_prob = self.fwd_probability.max(1e-300);
        let rev_prob = self.rev_probability.max(1e-300);

        format!(
            "{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t.\tVT={vt};CD={cd}\t\
GT:DP:AO:ER:TNC:PR:MFR:MFA:BFR:BFA:AMQR:AMQA:ABQR:ABQA:REDR:REDA:ISR:ISA:\
FWDP:REVP:LLE:SLE:REFC:AMPR:MFC:ARL:FWD:REV:TOT\t\
{gt}:{dp}:{ao}:{er:.3E}:{up}{rb}{dn}:{pr:.3E}:{mfr:.1}:{mfa:.1}:{bfr:.1}:{bfa:.1}:\
{amqr:.1}:{amqa:.1}:{abqr:.1}:{abqa:.1}:{redr:.1}:{reda:.1}:{isr:.1}:{isa:.1}:\
{fwdp:.3E}:{revp:.3E}:{lle:.3}:{sle:.1}:{refc:.1}:{ampr:.1}:{mfc:.1}:{arl:.1}:\
{fwd:.1}:{rev:.1}:{tot:.1}\n",
            chrom = self.contig,
            pos   = self.pos,
            ref   = self.reference,
            alt   = self.alt,
            qual  = self.score.round(),
            vt    = self.infer_variant_type(),
            cd    = cd,
            gt    = self.genotype,
            dp    = self.depth,
            ao    = self.alt_counts,
            er    = self.error_rate,
            up    = self.tnc.upstream_base as char,
            rb    = self.tnc.ref_base as char,
            dn    = self.tnc.downstream_base as char,
            pr    = prob,
            mfr   = self.mapq_filtered_ref,
            mfa   = self.mapq_filtered_alt,
            bfr   = self.bq_filtered_ref,
            bfa   = self.bq_filtered_alt,
            amqr  = self.average_ref_mapq,
            amqa  = self.average_alt_mapq,
            abqr  = self.average_ref_bq,
            abqa  = self.average_alt_bq,
            redr  = self.avg_ref_dist_from_read_end,
            reda  = self.avg_alt_dist_from_read_end,
            isr   = self.avg_ref_insert_size,
            isa   = self.avg_alt_insert_size,
            fwdp  = fwd_prob,
            revp  = rev_prob,
            lle   = self.large_local_entropy,
            sle   = self.small_local_entropy,
            refc  = self.read_end_filtered_count,
            ampr  = self.avg_mismatch_per_read,
            mfc   = self.mismatch_filtered_count,
            arl   = self.avg_read_length,
            fwd   = self.forward_strand_count_snps,
            rev   = self.reverse_strand_count_snps,
            tot   = self.both_strands_count_snps,
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
        let p_best = best_prob / all_probs_sum;
        let p_err = (1.0 - p_best).max(1e-300);
        let score = (-10.0 * p_err.log10()).min(999.0);
        Genotype { genotype: genotype.to_string(), score }
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
    Indel,
}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
/// Types of variant observations
///
/// # Variants
/// * `Snp` - Single nucleotide polymorphism
/// * `Insertion` - Insertion variant
/// * `Deletion` - Deletion variant
/// * `Ref` - Reference allele
/// * `Complex` - Complex variant
enum VariantObservation {
    Snp,
    Insertion,
    Deletion,
    Ref,
}

#[derive(Clone, Debug)]
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

    fn check_variant_type(&self) -> VariantObservation {
        if self.insertion_bases.is_empty() && self.deleted_bases.is_empty() {
            if self.ref_base != self.base {
                VariantObservation::Snp
            } else {
                VariantObservation::Ref
            }
        } else if !self.insertion_bases.is_empty() {
            VariantObservation::Insertion
        } else if !self.deleted_bases.is_empty() {
            VariantObservation::Deletion
        } else {
            panic!("Unexpected variant observed");
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
    if alt_candidates.iter().any(|bc| {
        matches!(
            bc.check_variant_type(),
            VariantObservation::Insertion | VariantObservation::Deletion
        )
    }) {
        return CallingDirective::Indel;
    }

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
/// Select candidates and counts based on calling directive
///
/// # Arguments
/// * `ref_base` - Reference base at the position
/// * `upstream_base` - Base upstream of the position
/// * `downstream_base` - Base downstream of the position
/// * `fwd_candidates` - Set of forward strand base candidates
/// * `fwd_counts` - Counts of forward strand base calls
/// * `rev_candidates` - Set of reverse strand base candidates
/// * `rev_counts` - Counts of reverse strand base calls
/// * `total_counts` - Total counts of base calls
/// # Returns
///
/// A tuple containing the selected candidates and their counts
fn select_candidates_and_counts(
    ref_base: char,
    upstream_base: char,
    downstream_base: char,
    fwd_candidates: &HashSet<BaseCall>,
    fwd_counts: &HashMap<BaseCall, usize>,
    rev_candidates: &HashSet<BaseCall>,
    rev_counts: &HashMap<BaseCall, usize>,
    total_counts: &HashMap<BaseCall, usize>,
    fwd_probabilities: &Vec<f64>,
    rev_probabilities: &Vec<f64>,
    total_probabilities: &Vec<f64>,
) -> (HashSet<BaseCall>, HashMap<BaseCall, usize>, Vec<f64>) {
    let directive =
        find_where_to_call_variants(ref_base, fwd_candidates, upstream_base, downstream_base);
    
    match directive {
        CallingDirective::ReferenceSiteOb | CallingDirective::DenovoSiteOb => {
            (rev_candidates.clone(), rev_counts.clone(), rev_probabilities.clone())
        }
        CallingDirective::ReferenceSiteOt | CallingDirective::DenovoSiteOt => {
            (fwd_candidates.clone(), fwd_counts.clone(), fwd_probabilities.clone())
        }
        CallingDirective::BothStrands | CallingDirective::Indel => {
            let intersection: HashSet<BaseCall> = fwd_candidates.intersection(rev_candidates).cloned().collect();
            (intersection, total_counts.clone(), total_probabilities.clone())
        }
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
##FORMAT=<ID=AO,Number=1,Type=Integer,Description=\"Alternate Allele Count\">\n\
##FORMAT=<ID=ER,Number=1,Type=Float,Description=\"Estimated Error Rate\">\n\
##FORMAT=<ID=TNC,Number=3,Type=String,Description=\"Trinucleotide Context (upstream,ref,downstream)\">\n\
##FORMAT=<ID=PR,Number=1,Type=Float,Description=\"Probability of the called genotype\">\n\
##FORMAT=<ID=MFR,Number=1,Type=Float,Description=\"Count of reference-supporting reads filtered by mapping quality\">\n\
##FORMAT=<ID=MFA,Number=1,Type=Float,Description=\"Count of alternate-supporting reads filtered by mapping quality\">\n\
##FORMAT=<ID=BFR,Number=1,Type=Float,Description=\"Count of reference-supporting reads filtered by base quality\">\n\
##FORMAT=<ID=BFA,Number=1,Type=Float,Description=\"Count of alternate-supporting reads filtered by base quality\">\n\
##FORMAT=<ID=AMQR,Number=1,Type=Float,Description=\"Average mapping quality of reads supporting the reference allele\">\n\
##FORMAT=<ID=AMQA,Number=1,Type=Float,Description=\"Average mapping quality of reads supporting the alternate allele\">\n\
##FORMAT=<ID=ABQR,Number=1,Type=Float,Description=\"Average base quality of reads supporting the reference allele\">\n\
##FORMAT=<ID=ABQA,Number=1,Type=Float,Description=\"Average base quality of reads supporting the alternate allele\">\n\
##FORMAT=<ID=REDR,Number=1,Type=Float,Description=\"Average distance from read end for reads supporting the reference allele\">\n\
##FORMAT=<ID=REDA,Number=1,Type=Float,Description=\"Average distance from read end for reads supporting the alternate allele\">\n\
##FORMAT=<ID=ISR,Number=1,Type=Float,Description=\"Average insert size for reads supporting the reference allele\">\n\
##FORMAT=<ID=ISA,Number=1,Type=Float,Description=\"Average insert size for reads supporting the alternate allele\">\n\
##FORMAT=<ID=FWDP,Number=1,Type=Float,Description=\"Probability of the called genotype based on forward strand reads only\">\n\
##FORMAT=<ID=REVP,Number=1,Type=Float,Description=\"Probability of the called genotype based on reverse strand reads only\">\n\
##FORMAT=<ID=LLE,Number=1,Type=Float,Description=\"Large local sequence entropy (50 bp on either side)\">\n\
##FORMAT=<ID=SLE,Number=1,Type=Float,Description=\"Small local sequence entropy (15 bp on either side)\">\n\
##FORMAT=<ID=REFC,Number=1,Type=Float,Description=\"Count of reads filtered due to proximity to read ends\">\n\
##FORMAT=<ID=AMPR,Number=1,Type=Float,Description=\"Average mismatches per read at the position\">\n\
##FORMAT=<ID=MFC,Number=1,Type=Float,Description=\"Count of reads filtered due to mismatches at the position\">\n\
##FORMAT=<ID=ARL,Number=1,Type=Float,Description=\"Average read length of reads covering the position\">\n\
##FORMAT=<ID=FWD,Number=1,Type=Float,Description=\"Forward counts\">\n\
##FORMAT=<ID=REV,Number=1,Type=Float,Description=\"Reverse counts\">\n\
##FORMAT=<ID=TOT,Number=1,Type=Float,Description=\"Both strand counts\">\n\
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
/// * `right_tail_pval` - Threshold for right-tail p-value
///
/// # Returns
/// Right-tail p-value
fn right_tail_binomial_pval(n: u64, k: u64, p: f64) -> f64 {
    let binom = Binomial::new(p, n).expect("Failed to create binomial dist");
    let cdf = binom.cdf(k - 1);
    1.0 - cdf
}
fn get_count_vec_candidates(
    counts: &HashMap<BaseCall, usize>,
    error_rate: f64,
) -> (HashSet<BaseCall>, Vec<f64>) {
    let total_depth = counts.values().sum::<usize>() as u64;
    let mut candidates = HashSet::new();
    let mut probabilities = Vec::new();

    for (basecall, &count) in counts {
        let variant = basecall.check_variant_type();
        let probability = right_tail_binomial_pval(total_depth, count as u64, error_rate);

        let keep = match variant {
            VariantObservation::Snp if basecall.base == 'N' || basecall.base == basecall.ref_base => false,
            VariantObservation::Insertion | VariantObservation::Deletion if basecall.base == 'N' => false,
            VariantObservation::Ref => false,
            _ => true,
        };

        if keep {
            candidates.insert(basecall.clone());
            probabilities.push(probability);
        }
    }

    (candidates, probabilities)
}

/// Assign genotype based on binomial probabilities
///
/// # Arguments
/// * `alt_counts` - Count of reads supporting the alternate allele
/// * `depth` - Total read depth at the position
/// * `error_rate` - Expected error rate
///
/// # Returns
/// A Genotype instance with assigned genotype and quality score
fn assign_genotype(alt_counts: usize, depth: usize, error_rate: f64) -> Genotype {
    let homo_ref_prob = Binomial::new(error_rate, depth as u64)
        .unwrap()
        .pmf(alt_counts as u64);
    let het_prob = Binomial::new(0.5, depth as u64)
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

fn infer_variant_type_from_alleles(reference: &str, alt: &str) -> &'static str {
    let rlen = reference.len();
    let alen = alt.len();
    match (rlen, alen) {
        (1, 1) => "SNP",
        (r, a) if r > 1 && a > 1 && r == a => "MNP",
        (r, 1) if r > 1 => "DEL",
        (1, a) if a > 1 => "INS",
        _ => "COMPLEX",
    }
}

#[derive(Debug)]
struct ModelInferenceConfig {
    model_path: String,
    threshold: f64,
    model_exists: bool,
}

#[derive(Debug, Clone)]
struct ModelFeatureInputs {
    depth: f64,
    alt_counts: f64,
    error_rate: f64,
    caller_probability: f64,
    mapq_filtered_ref: f64,
    mapq_filtered_alt: f64,
    bq_filtered_ref: f64,
    bq_filtered_alt: f64,
    average_ref_mapq: f64,
    average_alt_mapq: f64,
    average_ref_bq: f64,
    average_alt_bq: f64,
    avg_ref_dist: f64,
    avg_alt_dist: f64,
    avg_ref_ins: f64,
    avg_alt_ins: f64,
    fwd_probability: f64,
    rev_probability: f64,
    large_entropy: f64,
    small_entropy: f64,
    read_end_filtered_count: f64,
    avg_mismatch_per_read: f64,
    mismatch_filtered_count: f64,
    avg_read_length: f64,
    fwd_count: f64,
    rev_count: f64,
    total_count: f64,
    tnc_up: char,
    tnc_ref: char,
    tnc_down: char,
    vt: &'static str,
}

fn model_inference_config(model_path: &str) -> &'static ModelInferenceConfig {
    static CONFIG: OnceLock<ModelInferenceConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let model_path = model_path.to_string();
        let model_exists = Path::new(&model_path).exists();

        if model_exists {
            info!(
                "Detected ONNX model at {}. Inference hook is active and can be swapped to runtime execution.",
                model_path
            );
        } else {
            info!(
                "No ONNX model found at {}. Falling back to placeholder scoring.",
                model_path
            );
        }

        ModelInferenceConfig {
            model_path,
            threshold: MODEL_PROBABILITY_THRESHOLD,
            model_exists,
        }
    })
}

/// Build model input features in a stable order.
///
/// IMPORTANT: keep the order in sync with model training.
fn build_model_feature_vector(inputs: &ModelFeatureInputs) -> Vec<f32> {
    let af = if inputs.depth > 0.0 {
        inputs.alt_counts / inputs.depth
    } else {
        0.0
    };
    let mq_diff = inputs.average_alt_mapq - inputs.average_ref_mapq;
    let bq_diff = inputs.average_alt_bq - inputs.average_ref_bq;
    let red_diff = inputs.avg_alt_dist - inputs.avg_ref_dist;
    let is_diff = inputs.avg_alt_ins - inputs.avg_ref_ins;
    let strand_bias = (inputs.fwd_probability - inputs.rev_probability).abs();

    let mut values = HashMap::<&str, f64>::new();
    values.insert("DP", inputs.depth);
    values.insert("AO", inputs.alt_counts);
    values.insert("ER", inputs.error_rate);
    values.insert("PR", inputs.caller_probability);
    values.insert("MFR", inputs.mapq_filtered_ref);
    values.insert("MFA", inputs.mapq_filtered_alt);
    values.insert("BFR", inputs.bq_filtered_ref);
    values.insert("BFA", inputs.bq_filtered_alt);
    values.insert("AMQR", inputs.average_ref_mapq);
    values.insert("AMQA", inputs.average_alt_mapq);
    values.insert("ABQR", inputs.average_ref_bq);
    values.insert("ABQA", inputs.average_alt_bq);
    values.insert("REDR", inputs.avg_ref_dist);
    values.insert("REDA", inputs.avg_alt_dist);
    values.insert("ISR", inputs.avg_ref_ins);
    values.insert("ISA", inputs.avg_alt_ins);
    values.insert("FWDP", inputs.fwd_probability);
    values.insert("REVP", inputs.rev_probability);
    values.insert("LLE", inputs.large_entropy);
    values.insert("SLE", inputs.small_entropy);
    values.insert("REFC", inputs.read_end_filtered_count);
    values.insert("AMPR", inputs.avg_mismatch_per_read);
    values.insert("MFC", inputs.mismatch_filtered_count);
    values.insert("ARL", inputs.avg_read_length);
    values.insert("FWD", inputs.fwd_count);
    values.insert("REV", inputs.rev_count);
    values.insert("TOT", inputs.total_count);
    values.insert("AF", af);
    values.insert("MQ_diff", mq_diff);
    values.insert("BQ_diff", bq_diff);
    values.insert("RED_diff", red_diff);
    values.insert("IS_diff", is_diff);
    values.insert("strand_bias", strand_bias);

    let up = inputs.tnc_up.to_ascii_uppercase();
    let rf = inputs.tnc_ref.to_ascii_uppercase();
    let dn = inputs.tnc_down.to_ascii_uppercase();
    for base in MODEL_TNC_BASES {
        values.insert(
            match base {
                'A' => "TNC_up_A",
                'C' => "TNC_up_C",
                'G' => "TNC_up_G",
                'T' => "TNC_up_T",
                _ => "TNC_up_N",
            },
            if up == base { 1.0 } else { 0.0 },
        );
        values.insert(
            match base {
                'A' => "TNC_ref_A",
                'C' => "TNC_ref_C",
                'G' => "TNC_ref_G",
                'T' => "TNC_ref_T",
                _ => "TNC_ref_N",
            },
            if rf == base { 1.0 } else { 0.0 },
        );
        values.insert(
            match base {
                'A' => "TNC_down_A",
                'C' => "TNC_down_C",
                'G' => "TNC_down_G",
                'T' => "TNC_down_T",
                _ => "TNC_down_N",
            },
            if dn == base { 1.0 } else { 0.0 },
        );
    }

    let vt = inputs.vt;
    for vt_value in MODEL_VT_VALUES {
        values.insert(
            match vt_value {
                "COMPLEX" => "VT_COMPLEX",
                "DEL" => "VT_DEL",
                "INS" => "VT_INS",
                "MNP" => "VT_MNP",
                _ => "VT_SNP",
            },
            if vt == vt_value { 1.0 } else { 0.0 },
        );
    }

    MODEL_FEATURE_ORDER
        .iter()
        .map(|k| values.get(*k).copied().unwrap_or(0.0) as f32)
        .collect()
}

/// Placeholder ONNX inference hook.
///
/// Replace this function body with your ONNX runtime call. The signature and
/// call sites are already wired so you can plug inference in one place.
fn run_onnx_inference_placeholder(
    _model_path: &str,
    features: &[f32],
) -> Result<f64, Box<dyn std::error::Error>> {
    if features.len() < 2 {
        return Err("Feature vector must contain at least DP and AO".into());
    }

    // Temporary proxy score based on allele fraction.
    let depth = features[0] as f64;
    let alt_counts = features[1] as f64;
    let probability = if depth > 0.0 { alt_counts / depth } else { 0.0 };

    Ok(probability.clamp(0.0, 1.0))
}

fn model_probability_score(config: &ModelInferenceConfig, features: &[f32]) -> f64 {
    if config.model_exists {
        match run_onnx_inference_placeholder(&config.model_path, features) {
            Ok(probability) => probability,
            Err(_) => {
                // If runtime inference fails, do not drop calls silently.
                1.0
            }
        }
    } else {
        // No model file yet: keep baseline caller behavior (no ML filtering).
        1.0
    }
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

#[derive(Debug)]
/// Counts of basecalls in a pileup
struct PileupCounts {
    fwd: HashMap<BaseCall, usize>,
    rev: HashMap<BaseCall, usize>,
    total: HashMap<BaseCall, usize>,
}
/// Returns true if a slice has a repeated pattern of length n
/// at the start or end, with at least cutoff bases.
/// Return `true` if `sequence` contains a repeated unit of length `n` of at
/// least `cutoff` bases at the start or end.
fn has_repeat(sequence: &[u8], n: usize, cutoff: usize) -> bool {
    let len = sequence.len();
    if len < cutoff || n == 0 {
        return false;
    }

    let unit = &sequence[0..n];
    let start_ok = sequence[..cutoff].chunks(n).all(|chunk| chunk == unit);
    if start_ok {
        return true;
    }

    let tail_unit = &sequence[len - n..];
    let end_ok = sequence[len - cutoff..].chunks(n).all(|chunk| chunk == tail_unit);
    end_ok
}

/// Returns true if the read should be filtered out for INDEL calling
/// Filters reads with repeated sequences at the ends or soft-clipping
fn filter_indels(
    sequence: &[u8],
    record: &bam::Record,
    indel_filter_repeat_limit: usize,
    dinuc_cutoff: usize,
) -> bool {
    let homopolymer = has_repeat(sequence, 1, indel_filter_repeat_limit);
    let dinuc = has_repeat(sequence, 2, dinuc_cutoff);
    let soft_clipped = {
        for cigar in record.cigar().iter() {
            if let Cigar::SoftClip(_) = cigar {
                return true;
            }
        }
        false
    };
    homopolymer || dinuc || soft_clipped
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TrinucleotideContext {
    upstream_base: u8,
    ref_base: u8,
    downstream_base: u8,
}

impl TrinucleotideContext {
    fn new(upstream_base: u8, ref_base: u8, downstream_base: u8) -> Self {
        TrinucleotideContext { upstream_base, ref_base, downstream_base }
    }
}

/// Calculate Shannon entropy of a sequence
/// Returns 0 for empty sequences, and is based on the frequency of A, C, G, T
/// Non-ACGT characters are ignored in the calculation
/// The formula is: -sum(p_i * log2(p_i)) for each base i, where p_i is the frequency of base i in the sequence
/// The entropy is measured in bits, and higher values indicate more diversity in the sequence
/// The maximum entropy for a sequence of A, C, G, T is 2 bits (when all bases are equally represented)
/// For example, the sequence "ACGT" has an entropy of 2 bits, while "AAAA" has an entropy of 0 bits
fn shannon_entropy(sequence: &[u8]) -> f64 {
    if sequence.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 4];
    let mut valid = 0u32;
    for &base in sequence {
        match base {
            b'A' | b'a' => counts[0] += 1,
            b'C' | b'c' => counts[1] += 1,
            b'G' | b'g' => counts[2] += 1,
            b'T' | b't' => counts[3] += 1,
            _ => {} 
        }
        valid += 1;
    }
    if valid == 0 {
        return 0.0;
    }
    let n = valid as f64;
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// All per-position statistics returned by [`compute_pileup_counts`].
struct PileupStats {
    ref_dist_from_read_end: f64,
    alt_dist_from_read_end: f64,
    ref_insert_size_sum: f64,
    alt_insert_size_sum: f64,
    total_alt_counts: f64,
    total_ref_counts: f64,
    count_ref_mapq: f64,
    count_alt_mapq: f64,
    count_ref_bq: f64,
    count_alt_bq: f64,
    mapq_filtered_ref: f64,
    mapq_filtered_alt: f64,
    bq_filtered_ref: f64,
    bq_filtered_alt: f64,
    read_end_filtered_count_snps: f64,
    read_end_filtered_count_indels: f64,
    mismatch_filtered_count: f64,
    total_mismatches: f64,
    total_read_length: f64,
    indel_offset: u64,
}

/// Compute base call counts from a pileup
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
/// A Counts instance with extracted counts
#[allow(clippy::too_many_arguments)]
fn compute_pileup_counts(
    pileup: &Pileup,
    min_bq: usize,
    min_mapq: usize,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    ref_seq: &[u8],
    ref_pos: u32,
    stranded_read: &ReadNumber,
    pileup_counts: &mut PileupCounts,
    indel_filter_repeat_limit: usize,
    dinuc_cutoff: usize,
) -> PileupStats {
    pileup_counts.fwd.clear();
    pileup_counts.rev.clear();
    pileup_counts.total.clear();

    let mut stats = PileupStats {
        ref_dist_from_read_end: 0.0,
        alt_dist_from_read_end: 0.0,
        ref_insert_size_sum: 0.0,
        alt_insert_size_sum: 0.0,
        total_alt_counts: 0.0,
        total_ref_counts: 0.0,
        count_ref_mapq: 0.0,
        count_alt_mapq: 0.0,
        count_ref_bq: 0.0,
        count_alt_bq: 0.0,
        mapq_filtered_ref: 0.0,
        mapq_filtered_alt: 0.0,
        bq_filtered_ref: 0.0,
        bq_filtered_alt: 0.0,
        read_end_filtered_count_snps: 0.0,
        read_end_filtered_count_indels: 0.0,
        mismatch_filtered_count: 0.0,
        total_mismatches: 0.0,
        total_read_length: 0.0,
        indel_offset: 0,
    };

    for alignment in pileup.alignments() {
        let record = alignment.record();
        let mismatches = get_nm_tag(&record);

        if mismatches > max_mismatches {
            stats.mismatch_filtered_count += 1.0;
        }
        stats.total_mismatches += mismatches as f64;

        let qpos = match alignment.qpos() {
            Some(p) => p,
            None => continue,
        };

        if alignment.is_del() || alignment.is_refskip() {
            continue;
        }

        let base = record.seq().as_bytes()[qpos] as char;
        if base == 'N' {
            continue;
        }

        let qual = record.qual()[qpos];
        let mapq = record.mapq();
        let basecall = BaseCall::new(&alignment, ref_seq, ref_pos);
        let variant_type = basecall.check_variant_type();

        if qual < min_bq as u8 {
            if variant_type == VariantObservation::Ref {
                stats.bq_filtered_ref += 1.0;
            } else {
                stats.bq_filtered_alt += 1.0;
            }
            continue;
        }

        if mapq < min_mapq as u8 {
            if variant_type == VariantObservation::Ref {
                stats.mapq_filtered_ref += 1.0;
            } else {
                stats.mapq_filtered_alt += 1.0;
            }
            continue;
        }

        let is_ref = variant_type == VariantObservation::Ref;
        if is_ref {
            stats.total_ref_counts += 1.0;
            stats.count_ref_mapq += mapq as f64;
            stats.count_ref_bq += qual as f64;
            stats.ref_dist_from_read_end +=
                std::cmp::min(qpos, record.seq().len() - 1 - qpos) as f64;
            stats.ref_insert_size_sum += record.insert_size().unsigned_abs() as f64;
        } else {
            stats.total_alt_counts += 1.0;
            stats.count_alt_mapq += mapq as f64;
            stats.count_alt_bq += qual as f64;
            stats.alt_dist_from_read_end +=
                std::cmp::min(qpos, record.seq().len() - 1 - qpos) as f64;
            stats.alt_insert_size_sum += record.insert_size().unsigned_abs() as f64;
        }

        if record.is_secondary() || record.is_supplementary() || record.is_duplicate() {
            continue;
        }

        stats.total_read_length += record.seq().len() as f64;

        let read_len = record.seq().len();
        match variant_type {
            VariantObservation::Snp => {
                if qpos < end_of_read_cutoff || qpos >= read_len - end_of_read_cutoff {
                    stats.read_end_filtered_count_snps += 1.0;
                }
            }
            VariantObservation::Insertion | VariantObservation::Deletion => {
                if qpos < indel_end_of_read_cutoff || qpos >= read_len - indel_end_of_read_cutoff {
                    stats.read_end_filtered_count_indels += 1.0;
                }
            }
            _ => {}
        }

        // Strand assignment.
        let on_rev = (record.is_reverse() && is_stranded_read(&record, stranded_read))
            || (!record.is_reverse() && !is_stranded_read(&record, stranded_read));

        if on_rev {
            *pileup_counts.rev.entry(basecall.clone()).or_insert(0) += 1;
        } else {
            *pileup_counts.fwd.entry(basecall.clone()).or_insert(0) += 1;
        }
        *pileup_counts.total.entry(basecall.clone()).or_insert(0) += 1;

        if is_ref {
            let read_seq = record.seq().as_bytes();
            if filter_indels(&read_seq, &record, indel_filter_repeat_limit, dinuc_cutoff) {
                stats.indel_offset += 1;
            }
        }
    }

    stats
}

/// Distributes counts from a pileup map into SNP and INDEL maps
///
/// # Arguments
/// * `pileup_map` - The pileup counts map
/// * `snp_map` - The SNP counts map to populate
/// * `indel_map` - The INDEL counts map to populate
fn distribute_counts(
    pileup_map: &HashMap<BaseCall, usize>,
    snp_map: &mut HashMap<BaseCall, usize>,
    indel_map: &mut HashMap<BaseCall, usize>,
) {
    for (obs, count) in pileup_map {
        match obs.check_variant_type() {
            VariantObservation::Snp | VariantObservation::Ref => {
                snp_map.insert(obs.clone(), *count);
            }
            VariantObservation::Insertion | VariantObservation::Deletion => {
                indel_map.insert(obs.clone(), *count);
            }
        }
    }
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
    indel_filter_repeat_limit: usize,
    model_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting TVC workflow");
    validate_fai_and_bam(ref_path, bam_path)?;

    info!("Reading reference sequences");
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

    info!("Dividing genome into chunks and getting ready for parallel processing");

    let chunks: Vec<GenomeChunk> = get_genome_chunks(ref_path, chunk_size);

    let pb = ProgressBar::new(chunks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks processed",
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
                    indel_filter_repeat_limit,
                    model_path,
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
/// Compute trinucleotide context-specific error rates for a genome chunk
///
/// # Arguments
/// * `chunk` - The genome chunk to analyze
/// * `bam_path` - Path to the BAM file
/// * `ref_seq` - The reference sequence for the chunk
/// * `min_bq` - Minimum base quality
/// * `min_mapq` - Minimum mapping quality
/// * `min_depth` - Minimum read depth
/// * `end_of_read_cutoff` - End of read cutoff for SNPs
/// * `indel_end_of_read_cutoff` - End of read cutoff for indels
/// * `max_mismatches` - Maximum allowed mismatches in a read
/// * `error_rate` - Expected general error rate
/// * `stranded_read` - Which read is stranded (R1 or R2)
/// * `indel_filter_repeat_limit` - Number of bases for homopolymer/dinucleotide repeat filtering for indels
/// # Returns
/// A HashMap mapping each trinucleotide context to its estimated error rate
fn compute_tnc_error_rates(
    chunk: &GenomeChunk,
    bam_path: &str,
    ref_seq: &[u8],
    min_bq: usize,
    min_mapq: usize,
    min_depth: u32,
    end_of_read_cutoff: usize,
    indel_end_of_read_cutoff: usize,
    max_mismatches: u32,
    error_rate: f64,
    stranded_read: &ReadNumber,
    indel_filter_repeat_limit: usize,
) -> Result<HashMap<TrinucleotideContext, f64>, Box<dyn std::error::Error>> {
    // Pre-populate every possible TNC with zero counts.
    let bases = [b'A', b'C', b'G', b'T'];
    let mut tnc_counts: HashMap<TrinucleotideContext, (f64, f64)> = HashMap::new();
    for &upstream in &bases {
        for &ref_base in &bases {
            for &downstream in &bases {
                tnc_counts.insert(TrinucleotideContext::new(upstream, ref_base, downstream), (0.0, 0.0));
            }
        }
    }

    let mut bam = bam::IndexedReader::from_path(bam_path)?;
    let header = bam.header().to_owned();
    let tid = header.tid(chunk.contig.as_bytes()).ok_or("Contig not found in BAM header")?;
    bam.fetch((tid, chunk.start as i64, chunk.end as i64))?;

    let mut pileup_counts = PileupCounts {
        fwd: HashMap::with_capacity(8),
        rev: HashMap::with_capacity(8),
        total: HashMap::with_capacity(8),
    };

    let mut fwd_snps = HashMap::with_capacity(4);
    let mut rev_snps = HashMap::with_capacity(4);
    let mut fwd_indels = HashMap::with_capacity(4);
    let mut rev_indels = HashMap::with_capacity(4);
    let mut total_snps = HashMap::with_capacity(4);
    let mut total_indels = HashMap::with_capacity(4);

    for result in bam.pileup() {
        let pileup: Pileup = result?;
        let pos = pileup.pos();

        if pileup.depth() < min_depth {
            continue;
        }

        let ref_base = ref_seq[pos as usize];
        let dinuc_cutoff = if !indel_filter_repeat_limit.is_multiple_of(2) {
            indel_filter_repeat_limit + 1
        } else {
            indel_filter_repeat_limit
        };

        compute_pileup_counts(
            &pileup, min_bq, min_mapq, end_of_read_cutoff, indel_end_of_read_cutoff,
            max_mismatches, ref_seq, pos, stranded_read, &mut pileup_counts,
            indel_filter_repeat_limit, dinuc_cutoff,
        );

        fwd_snps.clear(); rev_snps.clear(); fwd_indels.clear();
        rev_indels.clear(); total_snps.clear(); total_indels.clear();

        distribute_counts(&pileup_counts.fwd,   &mut fwd_snps,   &mut fwd_indels);
        distribute_counts(&pileup_counts.rev,   &mut rev_snps,   &mut rev_indels);
        distribute_counts(&pileup_counts.total, &mut total_snps, &mut total_indels);

        let upstream   = if pos > 0                              { ref_seq[pos as usize - 1] } else { b'N' };
        let downstream = if pos < ref_seq.len() as u32 - 1      { ref_seq[pos as usize + 1] } else { b'N' };

        let (fwd_cands, fwd_probs) = get_count_vec_candidates(&fwd_snps, error_rate);
        let (rev_cands, rev_probs) = get_count_vec_candidates(&rev_snps, error_rate);
        let (_total_cands, total_probs) = get_count_vec_candidates(&total_snps, error_rate);

        let (counts_snps, ..) = {
            let (cands, counts, probs) = select_candidates_and_counts(
                ref_base as char, upstream as char, downstream as char,
                &fwd_cands, &fwd_snps, &rev_cands, &rev_snps, &total_snps,
                &fwd_probs, &rev_probs, &total_probs,
            );
            (counts, cands, probs)
        };

        let total_ref_snps: u64 = counts_snps
            .iter()
            .filter(|(k, _)| k.check_variant_type() == VariantObservation::Ref)
            .map(|(_, &v)| v as u64)
            .sum();
        let total_alt_snps: u64 = counts_snps
            .iter()
            .filter(|(k, _)| k.check_variant_type() == VariantObservation::Snp)
            .map(|(_, &v)| v as u64)
            .sum();

        let ctx = TrinucleotideContext::new(upstream, ref_base, downstream);
        let entry = tnc_counts.entry(ctx).or_insert((0.0, 0.0));
        entry.0 += total_alt_snps as f64;
        entry.1 += total_ref_snps as f64;
    }

    let tnc_error_rates = tnc_counts
        .into_iter()
        .map(|(ctx, (alt, ref_count))| {
            let total = alt + ref_count;
            let er = if total > 0.0 {
                let af = alt / total;
                if af > 0.0 && af < 1.0 { af } else { error_rate }
            } else {
                error_rate
            };
            (ctx, er)
        })
        .collect();

    Ok(tnc_error_rates)
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

/// Call all SNP and indel variants in one genome chunk.
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
    indel_filter_repeat_limit: usize,
    model_path: &str,
) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    let model_config = model_inference_config(model_path);

    let error_map = compute_tnc_error_rates(
        chunk, bam_path, ref_seq, min_bq, min_mapq, min_depth,
        end_of_read_cutoff, indel_end_of_read_cutoff, max_mismatches,
        error_rate, stranded_read, indel_filter_repeat_limit,
    )?;

    let mut bam = bam::IndexedReader::from_path(bam_path)?;
    let header = bam.header().to_owned();
    let tid = header.tid(chunk.contig.as_bytes()).ok_or("Contig not found in BAM header")?;
    bam.fetch((tid, chunk.start as i64, chunk.end as i64))?;

    let mut variants = Vec::new();
    let mut pileup_counts = PileupCounts {
        fwd: HashMap::with_capacity(8),
        rev: HashMap::with_capacity(8),
        total: HashMap::with_capacity(8),
    };

    let mut fwd_snps   = HashMap::with_capacity(4);
    let mut rev_snps   = HashMap::with_capacity(4);
    let mut fwd_indels = HashMap::with_capacity(4);
    let mut rev_indels = HashMap::with_capacity(4);
    let mut total_snps   = HashMap::with_capacity(4);
    let mut total_indels = HashMap::with_capacity(4);

    for result in bam.pileup() {
        let pileup: Pileup = result?;
        let tid = pileup.tid();
        let ref_name = std::str::from_utf8(header.tid2name(tid))?;
        let pos = pileup.pos();
        let ref_base = ref_seq[pos as usize];

        // if pileup.depth() < min_depth {
        //     continue;
        // }

        let dinuc_cutoff = if !indel_filter_repeat_limit.is_multiple_of(2) {
            indel_filter_repeat_limit + 1
        } else {
            indel_filter_repeat_limit
        };

        let s = compute_pileup_counts(
            &pileup, min_bq, min_mapq, end_of_read_cutoff, indel_end_of_read_cutoff,
            max_mismatches, ref_seq, pos, stranded_read, &mut pileup_counts,
            indel_filter_repeat_limit, dinuc_cutoff,
        );

        // Derived averages.
        let div = |num: f64, den: f64| if num > 0.0 && den > 0.0 { num / den } else { 0.0 };
        let average_ref_mapq = div(s.count_ref_mapq, s.total_ref_counts);
        let average_alt_mapq = div(s.count_alt_mapq, s.total_alt_counts);
        let average_ref_bq   = div(s.count_ref_bq, s.total_ref_counts);
        let average_alt_bq   = div(s.count_alt_bq, s.total_alt_counts);
        let avg_ref_dist     = div(s.ref_dist_from_read_end, s.total_ref_counts);
        let avg_alt_dist     = div(s.alt_dist_from_read_end, s.total_alt_counts);
        let avg_ref_ins      = div(s.ref_insert_size_sum, s.total_ref_counts);
        let avg_alt_ins      = div(s.alt_insert_size_sum, s.total_alt_counts);
        let total_reads      = s.total_ref_counts + s.total_alt_counts;
        let avg_mismatch     = div(s.total_mismatches, total_reads);
        let avg_read_length  = div(s.total_read_length, total_reads);

        fwd_snps.clear(); rev_snps.clear(); fwd_indels.clear();
        rev_indels.clear(); total_snps.clear(); total_indels.clear();

        distribute_counts(&pileup_counts.fwd,   &mut fwd_snps,   &mut fwd_indels);
        distribute_counts(&pileup_counts.rev,   &mut rev_snps,   &mut rev_indels);
        distribute_counts(&pileup_counts.total, &mut total_snps, &mut total_indels);

        let upstream   = if pos > 0                         { ref_seq[pos as usize - 1] } else { b'N' };
        let downstream = if pos < ref_seq.len() as u32 - 1 { ref_seq[pos as usize + 1] } else { b'N' };

        // Entropy.
        let large_flank = 50usize;
        let large_entropy = shannon_entropy(
            &ref_seq[(pos as usize).saturating_sub(large_flank)
                ..((pos as usize + large_flank + 1).min(ref_seq.len()))],
        );
        let small_flank = 15usize;
        let small_entropy = shannon_entropy(
            &ref_seq[(pos as usize).saturating_sub(small_flank)
                ..((pos as usize + small_flank + 1).min(ref_seq.len()))],
        );

        let ctx = TrinucleotideContext::new(upstream, ref_base, downstream);
        let tnc_er = error_map.get(&ctx).copied().unwrap_or(error_rate);

        let (fwd_cands, fwd_probs) = get_count_vec_candidates(&fwd_snps, tnc_er);
        let (rev_cands, rev_probs) = get_count_vec_candidates(&rev_snps, tnc_er);
        let (_total_cands_snps, total_probs_snps) = get_count_vec_candidates(&total_snps, tnc_er);

        let (fwd_indel_cands, fwd_indel_probs) = get_count_vec_candidates(&fwd_indels, tnc_er);
        let (rev_indel_cands, rev_indel_probs) = get_count_vec_candidates(&rev_indels, tnc_er);
        let (_total_cands_indels, total_probs_indels) = get_count_vec_candidates(&total_indels, tnc_er);

        let directive_snps = find_where_to_call_variants(
            ref_base as char, &fwd_cands, upstream as char, downstream as char,
        );

        let (candidate_snps, counts_snps, probs_snps) = select_candidates_and_counts(
            ref_base as char, upstream as char, downstream as char,
            &fwd_cands, &fwd_snps, &rev_cands, &rev_snps, &total_snps,
            &fwd_probs, &rev_probs, &total_probs_snps,
        );

        let (candidate_indels, counts_indels, probs_indels) = select_candidates_and_counts(
            ref_base as char, upstream as char, downstream as char,
            &fwd_indel_cands, &fwd_indels, &rev_indel_cands, &rev_indels, &total_indels,
            &fwd_indel_probs, &rev_indel_probs, &total_probs_indels,
        );

        let total_depth_snps   = counts_snps.values().sum::<usize>() as u64;
        let total_depth_indels = counts_indels.values().sum::<usize>() as u64;
        let total_depth        = total_depth_snps + total_depth_indels;
        let total_depth_filtered = total_depth.saturating_sub(s.indel_offset);

        let prob_snps   = probs_snps.iter().sum::<f64>();
        let prob_indels = probs_indels.iter().sum::<f64>();

        let fwd_prob_sum = fwd_probs.iter().sum::<f64>();
        let rev_prob_sum = rev_probs.iter().sum::<f64>();
        let combined = (fwd_prob_sum + rev_prob_sum).max(1e-10);
        let fwd_bias = fwd_prob_sum / combined;
        let rev_bias = rev_prob_sum / combined;

        let fwd_count_snps  = fwd_snps.values().sum::<usize>() as f64;
        let rev_count_snps  = rev_snps.values().sum::<usize>() as f64;
        let both_count_snps = total_snps.values().sum::<usize>() as f64;

        let fwd_indel_prob_sum = fwd_indel_probs.iter().sum::<f64>();
        let rev_indel_prob_sum = rev_indel_probs.iter().sum::<f64>();
        let combined_indels = (fwd_indel_prob_sum + rev_indel_prob_sum).max(1e-10);
        let fwd_bias_indels = fwd_indel_prob_sum / combined_indels;
        let rev_bias_indels = rev_indel_prob_sum / combined_indels;

        let fwd_count_indels  = fwd_indels.values().sum::<usize>() as f64;
        let rev_count_indels  = rev_indels.values().sum::<usize>() as f64;
        let both_count_indels = total_indels.values().sum::<usize>() as f64;

        // --- Emit SNP variants ---
        if !candidate_snps.is_empty() && total_depth_snps >= min_depth as u64 {
            for candidate in candidate_snps {
                let alt_counts = *counts_snps.get(&candidate).unwrap_or(&0);
                let ref_allele = candidate.get_reference_allele();
                let alt_allele = candidate.get_alternate_allele();
                let vt = infer_variant_type_from_alleles(&ref_allele, &alt_allele);
                let model_inputs = ModelFeatureInputs {
                    depth: total_depth as f64,
                    alt_counts: alt_counts as f64,
                    error_rate: tnc_er,
                    caller_probability: prob_snps,
                    mapq_filtered_ref: s.mapq_filtered_ref,
                    mapq_filtered_alt: s.mapq_filtered_alt,
                    bq_filtered_ref: s.bq_filtered_ref,
                    bq_filtered_alt: s.bq_filtered_alt,
                    average_ref_mapq,
                    average_alt_mapq,
                    average_ref_bq,
                    average_alt_bq,
                    avg_ref_dist,
                    avg_alt_dist,
                    avg_ref_ins,
                    avg_alt_ins,
                    fwd_probability: fwd_bias,
                    rev_probability: rev_bias,
                    large_entropy,
                    small_entropy,
                    read_end_filtered_count: s.read_end_filtered_count_snps,
                    avg_mismatch_per_read: avg_mismatch,
                    mismatch_filtered_count: s.mismatch_filtered_count,
                    avg_read_length,
                    fwd_count: fwd_count_snps,
                    rev_count: rev_count_snps,
                    total_count: both_count_snps,
                    tnc_up: upstream as char,
                    tnc_ref: ref_base as char,
                    tnc_down: downstream as char,
                    vt,
                };
                let model_features = build_model_feature_vector(&model_inputs);
                let model_probability = model_probability_score(model_config, &model_features);
                if model_probability <= model_config.threshold {
                    continue;
                }
                let genotype = assign_genotype(alt_counts, total_depth as usize, tnc_er);

                variants.push(Variant::new(
                    ref_name.to_string(), pos + 1,
                    ref_allele, alt_allele,
                    genotype.genotype, genotype.score,
                    total_depth as u32, alt_counts as u32,
                    directive_snps.clone(), tnc_er, ctx.clone(),
                    prob_snps, s.mapq_filtered_ref, s.mapq_filtered_alt,
                    s.bq_filtered_ref, s.bq_filtered_alt,
                    average_ref_mapq, average_alt_mapq, average_ref_bq, average_alt_bq,
                    avg_ref_dist, avg_alt_dist, avg_ref_ins, avg_alt_ins,
                    fwd_bias, rev_bias, large_entropy, small_entropy,
                    s.read_end_filtered_count_snps, avg_mismatch, s.mismatch_filtered_count,
                    avg_read_length, fwd_count_snps, rev_count_snps, both_count_snps,
                ));
            }
        }

        // --- Emit indel variants ---
        if !candidate_indels.is_empty() && total_depth_indels >= min_depth as u64 {
            for candidate in candidate_indels {
                let alt_counts = *counts_indels.get(&candidate).unwrap_or(&0);
                if alt_counts < min_ao as usize {
                    continue;
                }
                let ref_allele = candidate.get_reference_allele();
                let alt_allele = candidate.get_alternate_allele();
                let vt = infer_variant_type_from_alleles(&ref_allele, &alt_allele);
                let model_inputs = ModelFeatureInputs {
                    depth: total_depth_filtered as f64,
                    alt_counts: alt_counts as f64,
                    error_rate: tnc_er,
                    caller_probability: prob_indels,
                    mapq_filtered_ref: s.mapq_filtered_ref,
                    mapq_filtered_alt: s.mapq_filtered_alt,
                    bq_filtered_ref: s.bq_filtered_ref,
                    bq_filtered_alt: s.bq_filtered_alt,
                    average_ref_mapq,
                    average_alt_mapq,
                    average_ref_bq,
                    average_alt_bq,
                    avg_ref_dist,
                    avg_alt_dist,
                    avg_ref_ins,
                    avg_alt_ins,
                    fwd_probability: fwd_bias_indels,
                    rev_probability: rev_bias_indels,
                    large_entropy,
                    small_entropy,
                    read_end_filtered_count: s.read_end_filtered_count_indels,
                    avg_mismatch_per_read: avg_mismatch,
                    mismatch_filtered_count: s.mismatch_filtered_count,
                    avg_read_length,
                    fwd_count: fwd_count_indels,
                    rev_count: rev_count_indels,
                    total_count: both_count_indels,
                    tnc_up: upstream as char,
                    tnc_ref: ref_base as char,
                    tnc_down: downstream as char,
                    vt,
                };
                let model_features = build_model_feature_vector(&model_inputs);
                let model_probability = model_probability_score(model_config, &model_features);
                if model_probability <= model_config.threshold {
                    continue;
                }
                let genotype = assign_genotype(alt_counts, total_depth_filtered as usize, 0.05);

                variants.push(Variant::new(
                    ref_name.to_string(), pos + 1,
                    ref_allele, alt_allele,
                    genotype.genotype, genotype.score,
                    total_depth_filtered as u32, alt_counts as u32,
                    CallingDirective::BothStrands, tnc_er, ctx.clone(),
                    prob_indels, s.mapq_filtered_ref, s.mapq_filtered_alt,
                    s.bq_filtered_ref, s.bq_filtered_alt,
                    average_ref_mapq, average_alt_mapq, average_ref_bq, average_alt_bq,
                    avg_ref_dist, avg_alt_dist, avg_ref_ins, avg_alt_ins,
                    fwd_bias_indels, rev_bias_indels, large_entropy, small_entropy,
                    s.read_end_filtered_count_indels, avg_mismatch, s.mismatch_filtered_count,
                    avg_read_length, fwd_count_indels, rev_count_indels, both_count_indels,
                ));
            }
        }
    }

    Ok(variants)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::faidx;

    macro_rules! make_variant_test {
        ($fn_name:ident, $bam_file:expr, $pos:expr, $ref_base:expr, $alt_base:expr, $gt:expr, $stranded_read:expr) => {
            #[test]
            fn $fn_name() {
                let test_ref = "test_assets/chr11.fasta";
                let test_bam = concat!("test_assets/testing_bams/", $bam_file);

                let ref_reader = faidx::Reader::from_path(test_ref).expect("Failed to open FASTA");
                let contig = "chr11";
                let seq_len = ref_reader.fetch_seq_len(contig);
                let ref_seq: Vec<u8> = ref_reader
                    .fetch_seq(contig, 0, seq_len as usize)
                    .expect("Failed to fetch seq")
                    .iter()
                    .map(|b| b.to_ascii_uppercase())
                    .collect();

                let chunk = GenomeChunk::new(contig.to_string(), $pos, $pos + 1);
                let variants = call_variants(
                    &chunk, test_bam, &ref_seq,
                    20, 1, 1, 5, 20, 10, 1, 0.005, &$stranded_read, 3, "model.onnx",
                )
                .expect("call_variants failed");

                if variants.is_empty() {
                    println!("Warning: No variants called");
                }
                for v in &variants {
                    println!("{}", v.to_vcf());
                }

                let matching = variants
                    .iter()
                    .find(|v| v.pos == $pos)
                    .expect("Expected variant not found");

                assert_eq!(matching.contig,    contig,     "Chromosome mismatch");
                assert_eq!(matching.reference, $ref_base,  "REF mismatch");
                assert_eq!(matching.alt,       $alt_base,  "ALT mismatch");
            }
        };
    }

    make_variant_test!(test_both_strands_chr11_8198900_a_c_homo,        "both_strands_chr11_8198900_A_C_homo.bam",        8198900,   "A", "C",      "1/1", ReadNumber::R1);
    make_variant_test!(test_both_strands_chr11_8198951_t_a_het,         "both_strands_chr11_8198951_T_A_het.bam",         8198951,   "T", "A",      "0/1", ReadNumber::R1);
    make_variant_test!(test_denovo_ob_chr11_134755809_t_c_homo,         "denovo_ob_chr11_134755809_T_C_homo.bam",         134755809, "T", "C",      "1/1", ReadNumber::R1);
    make_variant_test!(test_denovo_ob_chr11_134911365_t_c_het,          "denovo_ob_chr11_134911365_T_C_het.bam",          134911365, "T", "C",      "0/1", ReadNumber::R1);
    make_variant_test!(test_short_hetero_del,                           "chr11:1160400-1160500_short_hetero_del.bam",     1160456,   "AC", "A",     "0/1", ReadNumber::R1);
    make_variant_test!(test_long_ins_hetero,                            "chr11:228150-228350_long_ins_hetero.bam",        228244,    "C", "CA",     "0/1", ReadNumber::R1);
    make_variant_test!(test_short_insertion_homo,                       "chr11:6586900-6587100_short_ins_homo.bam",       6586999,   "T", "TG",     "1/1", ReadNumber::R1);
    make_variant_test!(test_long_ins_homo,                              "chr11:5888900-5889100_long_ins_homo.bam",        5889008,   "C", "CTAGAG", "1/1", ReadNumber::R1);
    make_variant_test!(test_denovo_ot_chr11_134749303_a_g_het,          "denovo_ot_chr11_134749303_A_G_het.bam",          134749303, "A", "G",      "0/1", ReadNumber::R1);
    make_variant_test!(test_denovo_ot_chr11_134479860_a_g_homo,         "denovo_ot_chr11_134479860_A_G_homo.bam",         134479860, "A", "G",      "1/1", ReadNumber::R1);
    make_variant_test!(test_ref_ob_chr11_134012307_c_a_het,             "ref_ob_chr11_134012307_C_A_het.bam",             134012307, "C", "A",      "0/1", ReadNumber::R1);
    make_variant_test!(test_ref_ob_chr11_134610622_c_t_homo,            "ref_ob_chr11_134610622_C_T_homo.bam",            134610622, "C", "T",      "1/1", ReadNumber::R1);
    make_variant_test!(test_ref_ot_chr11_134473154_g_a_homo,            "ref_ot_chr11_134473154_G_A_homo.bam",            134473154, "G", "A",      "1/1", ReadNumber::R1);
    make_variant_test!(test_ref_ot_chr11_8195526_g_a_het,               "ref_ot_chr11_8195526_G_A_het.bam",               8195526,   "G", "A",      "0/1", ReadNumber::R1);

    fn load_ref_seq(contig: &str) -> Vec<u8> {
        let ref_reader =
            faidx::Reader::from_path("test_assets/chr11.fasta").expect("Failed to open FASTA");
        let seq_len = ref_reader.fetch_seq_len(contig);
        ref_reader
            .fetch_seq(contig, 0, seq_len as usize)
            .expect("Failed to fetch seq")
            .iter()
            .map(|b| b.to_ascii_uppercase())
            .collect()
    }

    #[test]
    fn test_methylation_site_no_variants() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.bam",
            &ref_seq, 20, 1, 1, 5, 20, 10, 1, 0.005, &ReadNumber::R1, 3, "model.onnx",
        )
        .expect("call_variants failed");

        let in_range: Vec<_> = variants.iter().filter(|v| v.pos >= 134755601 && v.pos <= 134755621).collect();
        assert!(in_range.is_empty(), "Expected no variants in methylation site BAM");
    }

    #[test]
    fn test_single_ended_reads() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.single_end.bam",
            &ref_seq, 20, 1, 1, 5, 20, 10, 1, 0.005, &ReadNumber::R1, 3, "model.onnx",
        )
        .expect("call_variants failed");

        let in_range: Vec<_> = variants.iter().filter(|v| v.pos >= 134755601 && v.pos <= 134755621).collect();
        assert!(in_range.is_empty(), "Expected no variants in single-ended methylation site BAM");
    }

    #[test]
    fn test_read_two_stranded() {
        let contig = "chr11";
        let ref_seq = load_ref_seq(contig);
        let chunk = GenomeChunk::new(contig.to_string(), 134755601, 134755621);

        let variants = call_variants(
            &chunk,
            "test_assets/testing_bams/methylation_site_chr11_134755601_134755621.bam",
            &ref_seq, 20, 1, 1, 5, 20, 10, 1, 0.005, &ReadNumber::R2, 3, "model.onnx",
        )
        .expect("call_variants failed");

        let in_range: Vec<_> = variants.iter().filter(|v| v.pos >= 134755601 && v.pos <= 134755621).collect();
        assert_eq!(
            in_range.len(), 2,
            "Since R2 was flipped the caller should emit 2 variants, got {}",
            in_range.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Indel filter unit tests (outside the cfg(test) module so they use the same
// helper types defined at crate level)
// ---------------------------------------------------------------------------

#[cfg(test)]
struct Qualities(Vec<u8>);

#[cfg(test)]
impl Qualities {
    fn from_bytes(bytes: Vec<u8>) -> Self {
        Qualities(bytes)
    }
}

#[test]
fn test_homopolymer_read_start() {
    let cigar = bam::record::CigarString::from(vec![Cigar::Match(7)]);
    let mut rec = bam::Record::new();
    let seq = b"AAATGCC";
    rec.set(b"r", Some(&cigar), seq, &Qualities::from_bytes(vec![255; 7]).0);
    assert!(filter_indels(seq, &rec, 3, 4));

    let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(6)]);
    let mut rec2 = bam::Record::new();
    let seq2 = b"AATGCC";
    rec2.set(b"r", Some(&cigar2), seq2, &Qualities::from_bytes(vec![255; 6]).0);
    assert!(!filter_indels(seq2, &rec2, 3, 4));
}

#[test]
fn test_homopolymer_read_end() {
    let cigar = bam::record::CigarString::from(vec![Cigar::Match(6)]);
    let mut rec = bam::Record::new();
    let seq = b"GCCTTT";
    rec.set(b"r", Some(&cigar), seq, &Qualities::from_bytes(vec![255; 6]).0);
    assert!(filter_indels(seq, &rec, 3, 4));

    let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(5)]);
    let mut rec2 = bam::Record::new();
    let seq2 = b"GCCTT";
    rec2.set(b"r", Some(&cigar2), seq2, &Qualities::from_bytes(vec![255; 5]).0);
    assert!(!filter_indels(seq2, &rec2, 3, 4));
}

#[test]
fn test_dinucleotide_read_start() {
    let cigar = bam::record::CigarString::from(vec![Cigar::Match(6)]);
    let mut rec = bam::Record::new();
    let seq = b"ATATGC";
    rec.set(b"r", Some(&cigar), seq, &Qualities::from_bytes(vec![255; 6]).0);
    assert!(filter_indels(seq, &rec, 3, 4));

    let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(6)]);
    let mut rec2 = bam::Record::new();
    let seq2 = b"ATCGTG";
    rec2.set(b"r", Some(&cigar2), seq2, &Qualities::from_bytes(vec![255; 6]).0);
    assert!(!filter_indels(seq2, &rec2, 3, 4));
}

#[test]
fn test_dinucleotide_read_end() {
    let cigar = bam::record::CigarString::from(vec![Cigar::Match(6)]);
    let mut rec = bam::Record::new();
    let seq = b"GCCTTT";
    rec.set(b"r", Some(&cigar), seq, &Qualities::from_bytes(vec![255; 6]).0);
    assert!(filter_indels(seq, &rec, 3, 4));

    let cigar2 = bam::record::CigarString::from(vec![Cigar::Match(6)]);
    let mut rec2 = bam::Record::new();
    let seq2 = b"GCCTTG";
    rec2.set(b"r", Some(&cigar2), seq2, &Qualities::from_bytes(vec![255; 6]).0);
    assert!(!filter_indels(seq2, &rec2, 3, 4));
}

#[test]
fn test_check_soft_clip() {
    let cigar_sc = bam::record::CigarString::from(vec![
        Cigar::SoftClip(5),
        Cigar::Match(10),
        Cigar::SoftClip(3),
    ]);
    let mut rec = bam::Record::new();
    let seq = b"ACGTACGTAC";
    let qual = Qualities::from_bytes(vec![255; 10]).0;
    rec.set(b"r", Some(&cigar_sc), seq, &qual);
    assert!(filter_indels(seq, &rec, 3, 4));

    let cigar_no_sc = bam::record::CigarString::from(vec![Cigar::Match(10)]);
    let mut rec2 = bam::Record::new();
    rec2.set(b"r", Some(&cigar_no_sc), seq, &qual);
    assert!(!filter_indels(seq, &rec2, 3, 4));
}