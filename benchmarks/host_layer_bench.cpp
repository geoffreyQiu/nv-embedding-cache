/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// Mirrored from benchmarks/single_layer_bench.cpp at 803e278.
// HostEmbeddingLayer + LinearHostTable only. Host buffers, no CUDA calls in this harness.
#include <argparse/argparse.hpp>
#include <host_embedding_layer.hpp>
#include <linear_host_table.hpp>
#include <thread_pool.hpp>
#include <logging.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <sched.h>
#include <unistd.h>

using IndexT = int64_t;
using Clock = std::chrono::steady_clock;
namespace {
struct Config {
  int64_t num_rows, embedding_dim, batch, hotness, num_layers, threads;
  int64_t num_steps, num_warmup_steps, latency_steps, threadpool_workers;
  int64_t task_size_threshold, hitrate_interval;
  int64_t key_buffer_steps;
  uint64_t seed;
  double alpha, max_latency_ms;
  bool disable_threadpool;
  std::string data_type, latency_keys, csv_filename, tag;
  std::string caller_cpus_text, worker_cpus_text;
  std::vector<int> caller_cpus, worker_cpus;
  size_t row_bytes, keys_per_step, table_bytes, key_steps, total_threads;
};
void require(bool ok, const std::string& message) {
  if (!ok) throw std::invalid_argument(message);
}
size_t product(size_t a, size_t b) {
  require(!b || a <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) / b,
          "size multiplication exceeds signed 64-bit range");
  return a * b;
}
double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b-a).count();
}
std::vector<int> cpu_list(const std::string& text) {
  std::vector<int> cpus;
  if (text.empty()) return cpus;
  std::istringstream stream(text);
  std::string item;
  while (std::getline(stream,item,',')) {
    size_t end{};
    const int cpu=std::stoi(item,&end);
    require(end==item.size() && cpu>=0 && cpu<1048576,"invalid CPU ID");
    cpus.push_back(cpu);
  }
  require(text.back()!=',' && std::set<int>(cpus.begin(),cpus.end()).size()==cpus.size(),
          "CPU lists must contain unique IDs and no empty entries");
  return cpus;
}
void pin_cpu(int cpu, pid_t tid=0) {
  // Dynamic masks also support Linux CPU IDs beyond CPU_SETSIZE.
  const size_t count=std::max<size_t>(cpu+1,sysconf(_SC_NPROCESSORS_CONF));
  const size_t bytes=CPU_ALLOC_SIZE(count);
  cpu_set_t* allowed=CPU_ALLOC(count);
  require(allowed!=nullptr,"cannot allocate CPU affinity mask");
  if (sched_getaffinity(tid,bytes,allowed)!=0 || !CPU_ISSET_S(cpu,bytes,allowed)) {
    CPU_FREE(allowed);
    throw std::runtime_error("requested CPU is outside the inherited affinity mask");
  }
  CPU_ZERO_S(bytes,allowed); CPU_SET_S(cpu,bytes,allowed);
  const int rc=sched_setaffinity(tid,bytes,allowed);
  CPU_FREE(allowed);
  require(rc==0,"cannot set thread CPU affinity");
}
std::set<pid_t> thread_ids() {
  std::set<pid_t> ids;
  for (const auto& entry:std::filesystem::directory_iterator("/proc/self/task"))
    ids.insert(static_cast<pid_t>(std::stol(entry.path().filename().string())));
  return ids;
}
class PowerLaw final {
 public:
  // alpha <= 0 samples uniformly; alpha == 1 is a pole of the inverse CDF so it is nudged.
  PowerLaw(double alpha, int64_t n) : n_(n), uniform_(alpha <= 0.0) {
    if (std::fabs(alpha - 1.0) < 1e-6) alpha = 1.0 + 1e-6;
    const double gamma = 1.0 - alpha;
    x0_ = 1.0;  // min rank 1 (shifted to 0 below)
    x1_sub_x0_ = std::pow(static_cast<double>(n) + 1.0, gamma) - x0_;
    exp_ = 1.0 / gamma;
  }
  template <typename G>
  int64_t operator()(G& gen) {
    if (uniform_) return static_cast<int64_t>(gen() % static_cast<uint64_t>(n_));
    const double x = dist_(gen);
    const double y = std::pow(x * x1_sub_x0_ + x0_, exp_);
    return std::clamp<int64_t>(static_cast<int64_t>(y) - 1, 0, n_ - 1);
  }

 private:
  int64_t n_;
  bool uniform_;
  double x0_{1.0}, x1_sub_x0_{0.0}, exp_{1.0};
  std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

// perm(x) = (a*x + b) mod n with gcd(a, n) = 1; memory free, shared by every thread.
class AffinePermutation final {
 public:
  AffinePermutation(int64_t n, uint64_t seed) : n_(static_cast<uint64_t>(std::max<int64_t>(n, 1))) {
    std::mt19937_64 g(seed ^ 0x9e3779b97f4a7c15ULL);
    b_ = g() % n_;
    a_ = (g() % n_) | 1ULL;
    while (std::gcd(a_, n_) != 1) {
      a_ += 2;
      if (a_ >= n_) a_ = 1;
    }
  }
  int64_t operator()(int64_t x) const {
    const unsigned __int128 t = static_cast<unsigned __int128>(a_) * static_cast<uint64_t>(x) + b_;
    return static_cast<int64_t>(static_cast<uint64_t>(t % n_));
  }

 private:
  uint64_t n_, a_{1}, b_{0};
};

class KeyGenerator final {
 public:
  KeyGenerator(const Config& cfg, uint64_t seed)
      : rng_(seed), law_(cfg.alpha, cfg.num_rows), perm_(cfg.num_rows, cfg.seed) {}
  void fill(IndexT* keys, int64_t n) {
    for (int64_t i = 0; i < n; ++i) keys[i] = perm_(law_(rng_));
  }

 private:
  std::mt19937_64 rng_;
  PowerLaw law_;
  AffinePermutation perm_;
};

class Barrier final {
 public:
  explicit Barrier(int64_t count) : threshold_(count), count_(count) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int64_t gen = generation_;
    if (--count_ == 0) {
      ++generation_;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  const int64_t threshold_;
  int64_t count_;
  int64_t generation_{0};
};

Config parse(int argc, char** argv) {
  argparse::ArgumentParser a("host_layer_bench",NVE_SOURCE_REV);
  a.add_description("CPU-only inference lookup through real NVE HostEmbeddingLayer; no pooling");
  a.add_argument("-nr", "--num_table_rows").default_value(int64_t(1000000)).scan<'i',int64_t>();
  a.add_argument("-ed", "--embedding_dim").default_value(int64_t(128)).scan<'i',int64_t>();
  a.add_argument("-dt", "--data_type").default_value(std::string("float32"));
  a.add_argument("-b", "--batch").default_value(int64_t(256)).scan<'i',int64_t>();
  a.add_argument("--hotness").default_value(int64_t(1024)).scan<'i',int64_t>();
  a.add_argument("-a", "--alpha").default_value(1.15).scan<'g',double>().help("power-law exponent; <= 0 is uniform");
  a.add_argument("-nl", "--num_layers").default_value(int64_t(1)).scan<'i',int64_t>();
  a.add_argument("-t", "--threads").default_value(int64_t(1)).scan<'i',int64_t>().help("concurrent inference callers per layer");
  a.add_argument("-ns", "--num_steps").default_value(int64_t(100)).scan<'i',int64_t>();
  a.add_argument("-nw", "--num_warmup_steps").default_value(int64_t(100)).scan<'i',int64_t>();
  a.add_argument("--latency_steps").default_value(int64_t(-1)).scan<'i',int64_t>().help("-1 = num_steps, 0 = skip");
  a.add_argument("--latency_keys").default_value(std::string("fresh")).help("fresh | replay");
  a.add_argument("--threadpool_workers").default_value(int64_t(1)).scan<'i',int64_t>().help("shared NVE workers; 0 = library default");
  a.add_argument("--disable_threadpool").default_value(false).implicit_value(true).help("run on inference callers instead of NVE workers");
  a.add_argument("--task_size_threshold").default_value(int64_t(1024)).scan<'i',int64_t>();
  a.add_argument("--hitrate_interval").default_value(int64_t(1)).scan<'i',int64_t>().help("0 disables sampling");
  a.add_argument("--max_latency_ms").default_value(0.0).scan<'g',double>();
  a.add_argument("--seed").default_value(uint64_t(42)).scan<'u',uint64_t>();
  a.add_argument("-csv", "--csv_filename").default_value(std::string(""));
  a.add_argument("--tag").default_value(std::string(""));
  a.add_argument("--key_buffer_steps").default_value(int64_t(512)).scan<'i',int64_t>()
      .help("maximum pre-generated batches per caller; replay cyclically for longer phases");
  a.add_argument("--caller_cpus").default_value(std::string("")).help("one logical CPU ID per caller");
  a.add_argument("--worker_cpus").default_value(std::string("")).help("one logical CPU ID per NVE worker");
  a.parse_args(argc, argv);
  Config c{};
#define GET(field, type, flag) c.field = a.get<type>(flag)
  GET(num_rows,int64_t,"--num_table_rows"); GET(embedding_dim,int64_t,"--embedding_dim");
  GET(batch,int64_t,"--batch"); GET(hotness,int64_t,"--hotness");
  GET(num_layers,int64_t,"--num_layers"); GET(threads,int64_t,"--threads");
  GET(num_steps,int64_t,"--num_steps"); GET(num_warmup_steps,int64_t,"--num_warmup_steps");
  GET(latency_steps,int64_t,"--latency_steps"); GET(threadpool_workers,int64_t,"--threadpool_workers");
  GET(task_size_threshold,int64_t,"--task_size_threshold"); GET(hitrate_interval,int64_t,"--hitrate_interval");
  GET(seed,uint64_t,"--seed"); GET(alpha,double,"--alpha"); GET(max_latency_ms,double,"--max_latency_ms");
  GET(disable_threadpool,bool,"--disable_threadpool"); GET(data_type,std::string,"--data_type");
  GET(latency_keys,std::string,"--latency_keys"); GET(csv_filename,std::string,"--csv_filename");
  GET(tag,std::string,"--tag");
  GET(key_buffer_steps,int64_t,"--key_buffer_steps");
  GET(caller_cpus_text,std::string,"--caller_cpus"); GET(worker_cpus_text,std::string,"--worker_cpus");
#undef GET
  require(c.num_rows>0 && c.num_rows <= (int64_t(1)<<53) && c.embedding_dim>0 && c.batch>0 && c.hotness>0,
          "table rows (<= 2^53), embedding_dim, batch and hotness must be positive");
  require(c.num_layers>0 && c.threads>0 && c.num_steps>0, "layers, threads and steps must be positive");
  require(c.num_warmup_steps>=0 && c.latency_steps>=-1 && c.threadpool_workers>=0 &&
          c.task_size_threshold>0 && c.hitrate_interval>=0, "invalid warmup/latency/worker/task/hitrate setting");
  require(std::isfinite(c.alpha) && std::isfinite(c.max_latency_ms) && c.max_latency_ms>=0, "alpha/latency budget must be finite; budget >= 0");
  require(c.latency_keys=="fresh" || c.latency_keys=="replay", "latency_keys must be fresh or replay");
  require(c.data_type=="float32" || c.data_type=="float16", "data_type must be float32 or float16");
  if (c.latency_steps==-1) c.latency_steps=c.num_steps;
  c.row_bytes=product(c.embedding_dim,c.data_type=="float32"?4:2);
  c.keys_per_step=product(c.batch,c.hotness);
  c.table_bytes=product(c.num_rows,c.row_bytes);
  c.total_threads=product(c.num_layers,c.threads);
  require(c.key_buffer_steps>0,"key_buffer_steps must be positive");
  c.key_steps=std::min(c.key_buffer_steps,c.latency_keys=="fresh" ? std::max(c.num_steps,c.latency_steps) : c.num_steps);
  c.caller_cpus=cpu_list(c.caller_cpus_text); c.worker_cpus=cpu_list(c.worker_cpus_text);
  require(c.caller_cpus.empty() || c.caller_cpus.size()==c.total_threads,"caller_cpus must have one CPU per caller");
  require(c.worker_cpus.empty() || (!c.disable_threadpool && c.threadpool_workers>0 &&
          c.worker_cpus.size()==static_cast<size_t>(c.threadpool_workers)),
          "worker_cpus requires an explicit matching pool size and enabled pool");
  const size_t key_bytes=product(product(c.keys_per_step,c.key_steps),sizeof(IndexT));
  const size_t out_bytes=product(c.keys_per_step,c.row_bytes);
  // Fail on overflow before allocation, including aggregate footprints.
  product(c.num_layers,c.table_bytes); product(c.total_threads,key_bytes);
  product(c.total_threads,out_bytes);
  // Reject exponents outside the upstream generator's finite numerical range.
  double alpha=c.alpha;
  if (std::fabs(alpha-1)<1e-6) alpha=1+1e-6;
  if (alpha>0) {
    const double span=std::pow(static_cast<double>(c.num_rows)+1,1-alpha);
    require(std::isfinite(span) && span>0 && std::isfinite(1/(1-alpha)),
            "alpha is outside the upstream power-law generator's numerical range");
  }
  return c;
}

// Deterministic payloads: byte-for-byte checks work for both fp32 and fp16 without conversion.
uint64_t payload(size_t row, size_t word, size_t layer) {
  uint64_t x=uint64_t(row)*0x9e3779b97f4a7c15ULL + word + uint64_t(layer)*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;
  return x^(x>>31);
}
void fill_row(uint8_t* dst, size_t bytes, size_t row, size_t layer) {
  for (size_t off=0; off<bytes; off+=8) {
    auto x=payload(row,off/8,layer);
    std::memcpy(dst+off,&x,std::min(size_t(8),bytes-off));
  }
}
struct Layer {
  std::vector<uint8_t> data;
  std::shared_ptr<nve::HostEmbeddingLayer<IndexT>> layer;
};
struct Result {
  Clock::time_point start,end,lat_start,lat_end;
  std::vector<double> latency, hits, lat_hits;
};
void verify(const Config& c, const Layer& l, nve::context_ptr_t& ctx) {
  const std::vector<IndexT> keys{0,c.num_rows/2,c.num_rows-1,-1,c.num_rows};
  std::vector<uint8_t> out(keys.size()*c.row_bytes,0xff);
  float hit=0;
  l.layer->lookup(ctx,keys.size(),keys.data(),out.data(),c.row_bytes,nullptr,nullptr,&hit);
  for (size_t i=0;i<keys.size();++i) {
    if (keys[i]>=0 && keys[i]<c.num_rows) {
      require(std::memcmp(out.data()+i*c.row_bytes,l.data.data()+keys[i]*c.row_bytes,c.row_bytes)==0,
              "preflight gather verification failed");
    } else {
      require(std::all_of(out.begin()+i*c.row_bytes,out.begin()+(i+1)*c.row_bytes,
                          [](uint8_t x){return x==0;}),"default row verification failed");
    }
  }
  require(std::abs(hit-0.6f)<1e-6f,"preflight hitrate verification failed");
}
void worker(const Config& c, const Layer& l, nve::context_ptr_t ctx, size_t idx,
            Barrier& barrier, Result& r) {
  std::vector<IndexT> keys(c.keys_per_step*c.key_steps);
  std::vector<IndexT> warm(c.keys_per_step);
  std::vector<uint8_t> out(c.keys_per_step*c.row_bytes);
  KeyGenerator gen(c,c.seed+1000*(idx+1));
  gen.fill(keys.data(),keys.size());
  for (int64_t s=0;s<c.num_warmup_steps;++s) {
    gen.fill(warm.data(),warm.size());
    l.layer->lookup(ctx,c.keys_per_step,warm.data(),out.data(),c.row_bytes,nullptr,nullptr,nullptr);
  }
  // One untimed full-size lookup also first-touches output/scratch when warmup=0.
  l.layer->lookup(ctx,c.keys_per_step,keys.data(),out.data(),c.row_bytes,nullptr,nullptr,nullptr);
  const size_t samples=c.hitrate_interval ? (c.num_steps-1)/c.hitrate_interval+1 : 0;
  r.hits.resize(samples); r.hits.clear();
  r.lat_hits.resize(c.hitrate_interval && c.latency_steps ? (c.latency_steps-1)/c.hitrate_interval+1 : 0); r.lat_hits.clear();
  r.latency.resize(c.latency_steps); // allocate and first-touch samples outside timing
  float hit=0;
  auto lookup=[&](size_t step, bool collect) {
    l.layer->lookup(ctx,c.keys_per_step,keys.data()+(step%c.key_steps)*c.keys_per_step,out.data(),
                    c.row_bytes,nullptr,nullptr,collect?&hit:nullptr);
  };
  barrier.wait();
  r.start=Clock::now();
  for (int64_t s=0;s<c.num_steps;++s) {
    const bool collect=c.hitrate_interval>0 && s%c.hitrate_interval==0;
    lookup(s,collect);
    if (collect) r.hits.push_back(hit);
  }
  r.end=Clock::now();
  // Prevent fresh key generation from interfering with any caller's throughput interval.
  barrier.wait();
  if (c.latency_keys=="fresh" && c.latency_steps) gen.fill(keys.data(),keys.size());
  barrier.wait();
  r.lat_start=Clock::now();
  for (int64_t s=0;s<c.latency_steps;++s) {
    const bool collect=c.hitrate_interval>0 && s%c.hitrate_interval==0;
    auto begin=Clock::now();
    lookup(c.latency_keys=="fresh"?s:s%c.num_steps,collect);
    const auto end=Clock::now();
    r.latency[s]=seconds(begin,end)*1000;
    if (collect) r.lat_hits.push_back(hit);
  }
  r.lat_end=Clock::now();
  barrier.wait(); // don't free buffers or verify while another caller is measuring
  const size_t last=c.latency_steps ? (c.latency_keys=="fresh"?c.latency_steps-1:(c.latency_steps-1)%c.num_steps) : c.num_steps-1;
  const auto* final_keys=keys.data()+(last%c.key_steps)*c.keys_per_step;
  for (size_t i=0;i<c.keys_per_step;++i) {
    require(std::memcmp(out.data()+i*c.row_bytes,l.data.data()+final_keys[i]*c.row_bytes,c.row_bytes)==0,
            "final measured output verification failed");
  }
}
std::string number(double x) { std::ostringstream s; s<<std::setprecision(12)<<x; return s.str(); }
std::string quote(const std::string& s) {
  if (s.find_first_of(",\"\r\n")==std::string::npos) return s;
  std::string out="\"";
  for(char x:s) { if(x=='\"') out+='\"'; out+=x; }
  return out+'\"';
}
using Metrics=std::vector<std::pair<std::string,std::string>>;
void csv(const std::string& path,const Metrics& m) {
  std::string header;
  for (const auto& v:m) { if(!header.empty()) header+=','; header+=quote(v.first); }
  std::ifstream in(path);
  std::string existing;
  const bool has_header=bool(std::getline(in,existing));
  require(!has_header || existing==header,"CSV header differs; choose a new output file");
  std::ofstream out(path,std::ios::app);
  require(bool(out),"cannot open CSV: "+path);
  if (!has_header) out<<header<<'\n';
  for(size_t i=0;i<m.size();++i) out<<(i?",":"")<<quote(m[i].second);
  out<<'\n'; out.flush(); require(bool(out),"CSV write failed");
}
double mean(const std::vector<double>& v) { return std::accumulate(v.begin(),v.end(),0.0)/static_cast<double>(v.size()); }
double percentile(const std::vector<double>& v,double p) {
  const double pos=p*static_cast<double>(v.size()-1);
  const auto i=static_cast<size_t>(pos);
  return v[i]+(v[std::min(i+1,v.size()-1)]-v[i])*(pos-static_cast<double>(i));
}
int run(const Config& c) {
  nve::GetGlobalLogger()->set_verbosity_level(nve::LogLevel_t::Error);
  const auto setup=Clock::now();
  nve::thread_pool_ptr_t pool;
  if(c.disable_threadpool) pool=nve::InlineThreadPool::instance();
  else {
    const auto before=c.worker_cpus.empty()?std::set<pid_t>{}:thread_ids();
    nve::SimpleThreadPoolConfig p; p.num_workers=c.threadpool_workers;
    pool=std::make_shared<nve::SimpleThreadPool>(p);
    if (!c.worker_cpus.empty()) {
      const auto after=thread_ids();
      std::vector<pid_t> workers;
      std::set_difference(after.begin(),after.end(),before.begin(),before.end(),std::back_inserter(workers));
      require(workers.size()==c.worker_cpus.size(),"cannot uniquely identify newly created NVE pool threads");
      for (size_t i=0;i<workers.size();++i) pin_cpu(c.worker_cpus[i],workers[i]);
    }
  }
  const double gib=1024.0*1024*1024;
  std::cout<<"HostLayer CPU inference; revision "<<NVE_SOURCE_REV<<"\n"
           <<c.num_layers<<" layers x "<<c.num_rows<<" rows x "<<c.embedding_dim<<" "<<c.data_type
           <<"; table GiB/layer "<<number(static_cast<double>(c.table_bytes)/gib)<<"\n"
           <<"Inference callers/layer "<<c.threads<<", shared NVE workers "<<pool->num_workers()
           <<", inline "<<c.disable_threadpool<<", alpha "<<c.alpha<<"\n"
           <<"Host key buffer GiB/caller "<<number(static_cast<double>(c.keys_per_step*c.key_steps*sizeof(IndexT))/gib)
           <<", output GiB/caller "<<number(static_cast<double>(c.keys_per_step*c.row_bytes)/gib)<<std::endl;
  std::vector<Layer> layers(c.num_layers);
  double prefill_s=0;
  for (size_t i=0;i<layers.size();++i) {
    auto& l=layers[i];
    const auto begin=Clock::now();
    l.data.resize(c.table_bytes);
    for(size_t row=0;row<static_cast<size_t>(c.num_rows);++row)
      fill_row(l.data.data()+row*c.row_bytes,c.row_bytes,row,i);
    prefill_s+=seconds(begin,Clock::now());
    nve::LinearHostTableConfig tc;
    tc.key_size=sizeof(IndexT); tc.num_rows=c.num_rows; tc.max_value_size=c.row_bytes;
    tc.value_dtype=c.data_type=="float32"?nve::DataType_t::Float32:nve::DataType_t::Float16;
    tc.emb_table=l.data.data(); tc.task_size_threshold=c.task_size_threshold;
    auto table=std::make_shared<nve::LinearHostTable<IndexT>>(tc);
    nve::HostEmbeddingLayer<IndexT>::Config lc;
    lc.layer_name="host_"+std::to_string(i); lc.disable_threadpool=c.disable_threadpool;
    lc.task_size_threshold=c.task_size_threshold; lc.default_embedding.resize(c.row_bytes,0);
    l.layer=std::make_shared<nve::HostEmbeddingLayer<IndexT>>(lc,table);
    auto ctx=l.layer->create_execution_context(nullptr,nullptr,pool,nullptr);
    verify(c,l,ctx);
  }
  std::vector<nve::context_ptr_t> contexts;
  for(size_t i=0;i<c.total_threads;++i)
    contexts.push_back(layers[i/c.threads].layer->create_execution_context(nullptr,nullptr,pool,nullptr));
  const double setup_s=seconds(setup,Clock::now());
  std::vector<Result> results(c.total_threads);
  Barrier barrier(c.total_threads);
  std::vector<std::thread> callers;
  for(size_t i=0;i<c.total_threads;++i) {
    callers.emplace_back([&,i] {
      try {
        if (!c.caller_cpus.empty()) pin_cpu(c.caller_cpus[i]);
        worker(c,layers[i/c.threads],contexts[i],i,barrier,results[i]);
      }
      catch(const std::exception& e) {
        // Fail fast: peers may be blocked at a phase barrier. Never emit partial metrics.
        std::cerr<<"ERROR: caller "<<i<<": "<<e.what()<<std::endl;
        std::_Exit(2);
      }
    });
  }
  for(auto& t:callers) t.join();
  auto window=[&](size_t first,size_t last) {
    auto start=results[first].start,end=results[first].end;
    for(size_t i=first;i<last;++i) { start=std::min(start,results[i].start); end=std::max(end,results[i].end); }
    return seconds(start,end);
  };
  const double elapsed=window(0,results.size());
  const double total_keys=static_cast<double>(c.total_threads)*static_cast<double>(c.num_steps)*static_cast<double>(c.keys_per_step);
  const double kps=total_keys/elapsed;
  std::vector<double> lat,hits,lat_hits;
  for(const auto& r:results) {
    lat.insert(lat.end(),r.latency.begin(),r.latency.end());
    hits.insert(hits.end(),r.hits.begin(),r.hits.end());
    lat_hits.insert(lat_hits.end(),r.lat_hits.begin(),r.lat_hits.end());
  }
  std::sort(lat.begin(),lat.end());
  const auto over=std::count_if(lat.begin(),lat.end(),[&](double x){return x>c.max_latency_ms;});
  std::cout<<"Time: "<<number(elapsed)<<", MKPS: "<<number(kps/1e6)
           <<", Algo BW: "<<number(kps*static_cast<double>(c.row_bytes)/1e9)<<" GB/s ("
           <<number(kps*static_cast<double>(c.row_bytes)/gib)<<" GiB/s)\n";
  for(size_t i=0;i<layers.size();++i)
    std::cout<<"Layer "<<i<<" MKPS: "<<number(static_cast<double>(c.threads)*static_cast<double>(c.num_steps)*static_cast<double>(c.keys_per_step)/window(i*c.threads,(i+1)*c.threads)/1e6)<<"\n";
  if(!hits.empty()) std::cout<<"Host hitrate: "<<mean(hits)<<" over "<<hits.size()<<" samples\n";
  if(!lat.empty()) {
    std::cout<<"Latency ms ("<<lat.size()<<" samples): mean "<<number(mean(lat))
             <<", p50 "<<number(percentile(lat,.5))<<", p95 "<<number(percentile(lat,.95))
             <<", p99 "<<number(percentile(lat,.99))<<", max "<<number(lat.back())<<"\n";
    if(c.max_latency_ms>0) std::cout<<"Above "<<c.max_latency_ms<<" ms: "<<over<<"\n";
  }
  struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
  char host[256]{}; gethostname(host,sizeof(host)-1);
  Metrics m;
  auto add=[&](const std::string& name,const auto& value) { std::ostringstream s; s<<std::setprecision(12)<<value; m.emplace_back(name,s.str()); };
  add("timestamp_unix_s",std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  add("hostname",host); add("nve_revision",NVE_SOURCE_REV); add("tag",c.tag);
  add("layer","host"); add("device","cpu"); add("data_type",c.data_type);
  add("embedding_dim",c.embedding_dim); add("num_table_rows",c.num_rows); add("batch",c.batch);
  add("hotness",c.hotness); add("alpha",c.alpha); add("seed",c.seed); add("num_layers",c.num_layers);
  add("threads",c.threads); add("threadpool_workers_requested",c.threadpool_workers);
  add("threadpool_workers_effective",pool->num_workers()); add("disable_threadpool",c.disable_threadpool);
  add("task_size_threshold",c.task_size_threshold); add("num_steps",c.num_steps);
  add("num_warmup_steps",c.num_warmup_steps); add("latency_keys",c.latency_keys);
  add("hitrate_interval",c.hitrate_interval); add("table_bytes_per_layer",c.table_bytes);
  add("key_bytes_per_caller",c.keys_per_step*c.key_steps*sizeof(IndexT));
  add("output_bytes_per_caller",c.keys_per_step*c.row_bytes);
  add("inference_time_s",elapsed); add("mkps",kps/1e6);
  add("gbps",kps*static_cast<double>(c.row_bytes)/1e9); add("gibps",kps*static_cast<double>(c.row_bytes)/gib);
  auto hit_metrics=[&](const std::string& prefix,const std::vector<double>& v) {
    add(prefix+"samples",v.size()); add(prefix+"mean",v.empty()?"":number(mean(v)));
    add(prefix+"min",v.empty()?"":number(*std::min_element(v.begin(),v.end())));
    add(prefix+"max",v.empty()?"":number(*std::max_element(v.begin(),v.end())));
  };
  hit_metrics("host_hitrate_",hits); hit_metrics("lat_host_hitrate_",lat_hits);
  add("lat_steps",lat.size()); add("lat_mean_ms",lat.empty()?"":number(mean(lat)));
  for(const auto& p:std::vector<std::pair<std::string,double>>{{"p50",.5},{"p95",.95},{"p99",.99}})
    add("lat_"+p.first+"_ms",lat.empty()?"":number(percentile(lat,p.second)));
  add("lat_max_ms",lat.empty()?"":number(lat.back())); add("max_latency_ms",c.max_latency_ms);
  add("lat_over_budget",c.max_latency_ms>0?std::to_string(over):"");
  add("setup_s",setup_s); add("prefill_s",prefill_s); add("peak_rss_kib",usage.ru_maxrss);
  add("verification","passed");
  auto lat_start=results.front().lat_start,lat_end=results.front().lat_end;
  for(const auto& r:results) { lat_start=std::min(lat_start,r.lat_start); lat_end=std::max(lat_end,r.lat_end); }
  add("latency_time_s",seconds(lat_start,lat_end)); add("key_buffer_steps_effective",c.key_steps);
  add("caller_cpus",c.caller_cpus_text); add("worker_cpus",c.worker_cpus_text);
  std::cout<<"Latency phase time: "<<seconds(lat_start,lat_end)<<" s\n";
  if(!c.csv_filename.empty()) csv(c.csv_filename,m);
  std::cout<<"Verification: passed (preflight, missing keys, final measured output)\n";
  return 0;
}
} // namespace
int main(int argc,char** argv) {
  try { return run(parse(argc,argv)); }
  catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
}
