// simd_fuzzy_deduplicationv7_unsorted_multi.cpp
// -----------------------------------------------------------
//  Build:
//  g++-13 -O3 -march=native -flto -fopenmp -std=c++17 -shared -fPIC \
//         simd_fuzzy_deduplicationv7_unsorted_multi.cpp MurmurHash3.cpp \
//         -o simd_fuzzy_deduplicationv7_unsorted_multi$(python3-config --extension-suffix)
//
//  Requires: pybind11 ≥ 2.6, OpenMP, MurmurHash3.cpp/.h in include path
// -----------------------------------------------------------
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>          // <- add this line
#include "MurmurHash3.h"

namespace py = pybind11;




/* ───────────────────── SIMD intersection counter ───────────────────── */
/* Works for any H multiple of 4 (AVX2) or 8 (AVX-512)                   */
#if defined(__AVX512F__)
inline std::pair<size_t,size_t>
simd_compute_intersection_union(const uint64_t* a,
                                const uint64_t* b,
                                size_t n) noexcept
{
    if (!n) return {0,0};
    constexpr size_t V = 8;                    // 8×64-bit / reg
    size_t inter = 0, i = 0;

    for (; i + V <= n; i += V) {
        __m512i v1 = _mm512_loadu_si512(a + i);
        __m512i v2 = _mm512_loadu_si512(b + i);
        __mmask8 k = _mm512_cmpeq_epi64_mask(v1, v2);
        inter     += _mm_popcnt_u64(k);        // one bit per lane
    }
    for (; i < n; ++i) inter += (a[i] == b[i]);
    return {inter, n};
}
#else   /* AVX2 fallback */
inline std::pair<size_t,size_t>
simd_compute_intersection_union(const uint64_t* a,
                                const uint64_t* b,
                                size_t n) noexcept
{
    if (!n) return {0,0};
    constexpr size_t V = 4;                    // 4×64-bit / reg
    size_t inter = 0, i = 0;

    for (; i + V <= n; i += V) {
        __m256i v1  = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i v2  = _mm256_loadu_si256((__m256i const*)(b + i));
        __m256i cmp = _mm256_cmpeq_epi64(v1, v2);
        uint32_t m  = _mm256_movemask_epi8(cmp);   // 32 bits
        inter      += __builtin_popcount(m) >> 3;  // ÷8 → lanes
    }
    for (; i < n; ++i) inter += (a[i] == b[i]);
    return {inter, n};
}
#endif


/* --------------------------------------------------------------
   simd_run_starts_pairs
   Given a vector of (hash, doc) pairs *sorted by hash*,
   fill `out` with the indices that start a new run of equal hashes.
   The first run always starts at 0, and we push a sentinel `n`
   as the last element so callers can loop `for(r=0;r+1<starts.size();++r)`.
-----------------------------------------------------------------*/
inline void simd_run_starts_pairs(
        const std::vector<std::pair<uint64_t,size_t>>& v,
        std::vector<size_t>& out)
{
    const size_t n = v.size();
    if (n == 0) return;

#if defined(__AVX512F__)
    constexpr size_t V = 8;
    size_t i = 1;                         // index we compare against v[i-1]
    for (; i + V <= n; i += V) {
        __m512i cur = _mm512_loadu_si512(&v[i]);
        __m512i prv = _mm512_loadu_si512(&v[i-1]);
        __mmask8 m  = _mm512_cmpeq_epi64_mask(cur, prv);
        uint8_t diff = static_cast<uint8_t>(~m);      // 1-bits → boundaries
        while (diff) {
            uint8_t tz = __builtin_ctz(diff);
            out.push_back(i + tz);
            diff &= diff - 1;
        }
    }
#elif defined(__AVX2__)
    constexpr size_t V = 4;
    size_t i = 1;
    for (; i + V <= n; i += V) {
        __m256i cur = _mm256_loadu_si256((__m256i const*)(&v[i]));
        __m256i prv = _mm256_loadu_si256((__m256i const*)(&v[i-1]));
        __m256i cmp = _mm256_cmpeq_epi64(cur, prv);
        uint32_t mask = _mm256_movemask_epi8(cmp);     // 32 bits: 8 per lane
        /* Bits 0,8,16,24 correspond to lane equality; invert to get diff */
        uint8_t diff = static_cast<uint8_t>(~mask) & 0x11;
        if (diff & 0x01) out.push_back(i    );
        if (diff & 0x10) out.push_back(i + 1);
        if (diff & 0x01<<4) out.push_back(i + 2);
        if (diff & 0x10<<4) out.push_back(i + 3);
    }
#else
    size_t i = 1;     // scalar fallback will handle everything
#endif
    /* scalar tail (and the whole array if no AVX) */
    for (; i < n; ++i)
        if (v[i].first != v[i-1].first)
            out.push_back(i);

    out.insert(out.begin(), 0);   // first run starts at 0
    out.push_back(n);             // sentinel
}



inline void avx2_run_starts_pairs(
        const std::vector<std::pair<uint64_t,size_t>>& v,
        std::vector<size_t>& out)
{
    const size_t n = v.size();
    if (n == 0) return;

    constexpr size_t V = 4;   // 4×uint64_t
    const uint64_t* base = reinterpret_cast<const uint64_t*>(v.data());
                              // interleave (hash,doc) so step = 2
    out.push_back(0);

    size_t i = 1;
    for (; i + V <= n; i += V) {
        /* load hash components only:   base[2*(i+k)]  */
        __m256i cur = _mm256_set_epi64x(base[2*(i+3)],
                                        base[2*(i+2)],
                                        base[2*(i+1)],
                                        base[2*(i+0)]);
        __m256i prv = _mm256_set_epi64x(base[2*(i+3) - 2],
                                        base[2*(i+2) - 2],
                                        base[2*(i+1) - 2],
                                        base[2*(i+0) - 2]);
        __m256i cmp = _mm256_cmpeq_epi64(cur, prv);
        uint32_t m  = _mm256_movemask_epi8(cmp);   // 8 bits / lane

        /* lane k boundary if lane-cmp != all-ones */
        if ((m &   0x000000FF) != 0x000000FF) out.push_back(i    );
        if ((m &   0x0000FF00) != 0x0000FF00) out.push_back(i + 1);
        if ((m &   0x00FF0000) != 0x00FF0000) out.push_back(i + 2);
        if ((m &   0xFF000000) != 0xFF000000) out.push_back(i + 3);
    }
    for (; i < n; ++i)
        if (v[i].first != v[i-1].first) out.push_back(i);

    out.push_back(n);     // sentinel
}



/* ──────────────────────  Phase E (pairwise)  ─────────────────────── */
static std::vector<std::vector<size_t>>
pairwise_jaccard(const std::vector<const uint64_t*>& rows,            // ★
                 size_t                               H,
                 const std::vector<std::pair<uint32_t,std::vector<size_t>>>& buckets,
                 double                               thr,
                 /* legacy debug containers */
                 std::vector<size_t>&              doc_ids_list,
                 std::vector<std::vector<size_t>>& docs_to_remove_list,
                 std::vector<size_t>&              len_of_docs2remove_list,
                 std::vector<size_t>&              docs_to_remove_all)
{
#pragma omp parallel
    {
        std::vector<size_t>              local_leaders;
        std::vector<std::vector<size_t>> local_dups;
        std::vector<size_t>              local_dup_flat;
        std::vector<size_t>              local_dup_len;

#pragma omp for schedule(static)
        for (size_t k = 0; k < buckets.size(); ++k) {
            const auto& ids = buckets[k].second;
            if (ids.size() <= 1) continue;

            std::vector<size_t> work(ids.begin(), ids.end());
            while (work.size() > 1) {
                const size_t pivot = work.front();
                const uint64_t* mh0 = rows[pivot];
                std::vector<size_t> to_remove, survivors;

                for (size_t i = 1; i < work.size(); ++i) {
                    const size_t cand = work[i];
                    const uint64_t* mh1 = rows[cand];
                    auto [inter, uni] =
                        simd_compute_intersection_union(mh0, mh1, H);
                    if (static_cast<double>(inter) / uni >= thr) {
                        to_remove.push_back(cand);
                        local_dup_flat.push_back(cand);
                    } else {
                        survivors.push_back(cand);
                    }
                }
                if (!to_remove.empty()) {
                    local_leaders.push_back(pivot);
                    local_dups.emplace_back(std::move(to_remove));
                    local_dup_len.push_back(local_dups.back().size());
                }
                work.swap(survivors);
            }
        }
#pragma omp critical
        {
            // doc_ids_list.insert(doc_ids_list.end(),
            //                      local_leaders.begin(), local_leaders.end());
            docs_to_remove_list.insert(docs_to_remove_list.end(),
                                       local_dups.begin(), local_dups.end());
            // len_of_docs2remove_list.insert(len_of_docs2remove_list.end(),
            //                                local_dup_len.begin(), local_dup_len.end());
            // docs_to_remove_all.insert(docs_to_remove_all.end(),
            //                            local_dup_flat.begin(), local_dup_flat.end());
        }
    }
    return docs_to_remove_list;
}

/* ─────────── Core LSH routine – rows = vector<const uint64_t*> ────────── */
static std::vector<std::vector<size_t>>
compute_jaccard_lsh_fast(const std::vector<const uint64_t*>& rows,    // ★
                         size_t                               H,
                         double                               thr,
                         size_t                               band_size)
{
    const size_t D = rows.size();
    if (!D) throw std::runtime_error("Input MinHash array is empty.");
    if (band_size == 0 || H % band_size)
        throw std::runtime_error("Invalid band size");

    const size_t B      = band_size;
    const size_t H_perB = H / B;
    const int    T      = omp_get_max_threads();

    /* ---------- Phase B: per-thread bucket fill (unchanged) ----------- */
    std::vector<std::vector<std::vector<std::pair<size_t,size_t>>>> tk(
        T, std::vector<std::vector<std::pair<size_t,size_t>>>(B));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local  = tk[tid];

#pragma omp for schedule(static)
        for (size_t doc = 0; doc < D; ++doc) {
            const uint64_t* sig = rows[doc];
            for (size_t b = 0; b < B; ++b) {
                uint64_t h[2];
                MurmurHash3_x64_128(sig + b * H_perB,
                                    H_perB * sizeof(uint64_t),
                                    42, h);
                local[b].emplace_back(h[0], doc);
            }
        }
    }




/* ---------- Phase C + D: gather + sort + SIMD run-scan -------------- */
std::vector<std::pair<uint32_t,std::vector<size_t>>> valid_buckets;  // OUTSIDE
#pragma omp parallel
{
    std::vector<std::pair<uint32_t,std::vector<size_t>>> local_valid;

#pragma omp for schedule(dynamic,8)
    for (size_t b = 0; b < B; ++b) {

        /* gather */
        std::vector<std::pair<uint64_t,size_t>> entries;
        size_t est = 0;
        for (int t = 0; t < T; ++t) est += tk[t][b].size();
        entries.reserve(est);
        for (int t = 0; t < T; ++t)
            entries.insert(entries.end(), tk[t][b].begin(), tk[t][b].end());

        if (entries.size() < 2) continue;

        std::sort(entries.begin(), entries.end(),
                  [](auto& x, auto& y){ return x.first < y.first; });

        /* SIMD run boundaries */
        // std::vector<size_t> starts;
        // simd_run_starts_pairs(entries, starts);

        std::vector<size_t> starts;
        avx2_run_starts_pairs(entries, starts);


        for (size_t r = 0; r + 1 < starts.size(); ++r) {
            size_t i = starts[r], j = starts[r+1];
            if (j - i <= 1) continue;

            std::vector<size_t> ids;
            ids.reserve(j - i);
            for (size_t k = i; k < j; ++k)
                ids.push_back(entries[k].second);

            std::sort(ids.begin(), ids.end());   // stable pivot
            local_valid.emplace_back(
                    static_cast<uint32_t>(entries[i].first),
                    std::move(ids));
        }
    }   /* end band loop */

#pragma omp critical
    valid_buckets.insert(valid_buckets.end(),
                         local_valid.begin(), local_valid.end());
}       /* end omp parallel */



    /* ---------- Phase E: pairwise (unchanged) ------------------------- */
    std::vector<size_t>              doc_ids_list;
    std::vector<std::vector<size_t>> docs_to_remove_list;
    std::vector<size_t>              len_of_docs2remove_list;
    std::vector<size_t>              docs_to_remove_all;

    pairwise_jaccard(rows, H, valid_buckets, thr,
                     doc_ids_list, docs_to_remove_list,
                     len_of_docs2remove_list, docs_to_remove_all);



    return docs_to_remove_list;
}

/* ─────────────── Python wrapper – zero-copy Phase A ───────────────── */
static std::vector<std::vector<size_t>>
simd_fuzzy_deduplicationv7_unsorted_multi7A_V1(py::array_t<uint64_t,
                           py::array::c_style | py::array::forcecast> minhashes,
               double  threshold,
               size_t  num_bands,
               size_t  num_threads = 0)
{
    if (num_threads)
        omp_set_num_threads(static_cast<int>(num_threads));

    const auto mh = minhashes.unchecked<2>();
    const size_t D = mh.shape(0);
    const size_t H = mh.shape(1);

    /* Phase A: build zero-copy pointer vector */
    std::vector<const uint64_t*> rows(D);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < D; ++i)
        rows[i] = &mh(i, 0);

    py::gil_scoped_release release;
    return compute_jaccard_lsh_fast(rows, H, threshold, num_bands);
}

/* ──────────────────────  PyBind11 glue  ──────────────────────────── */
PYBIND11_MODULE(simd_fuzzy_deduplicationv7_unsorted_multi7A_V1, m)
{
    m.doc() = "SIMD + OpenMP fuzzy deduplication (v7_V1, zero-copy only)";

    m.def("simd_fuzzy_deduplicationv7_unsorted_multi7A_V1",
          &simd_fuzzy_deduplicationv7_unsorted_multi7A_V1,
          py::arg("minhashes"),
          py::arg("threshold"),
          py::arg("num_bands"),
          py::arg("num_threads") = 0,
          R"pbdoc(
MinHash-based fuzzy deduplication with zero-copy row pointers.
All legacy debug counters are printed unchanged, but the initial
vector-of-vector deep copy is removed, shaving 1–2 seconds off a
3 M × 256 run on 32 cores.
)pbdoc");
}