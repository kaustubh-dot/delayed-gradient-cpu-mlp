// C++17 reference reconstruction from the supplied project design.
// One message is one example; this program does not accumulate mini-batches.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using Clock = std::chrono::steady_clock;
double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
constexpr int D=784, H=32, C=10;
using Input=std::array<double,D>;
using Hidden=std::array<double,H>;
using Logits=std::array<double,C>;

template<class T> class Queue {
    std::mutex mu;
    std::condition_variable readable, writable;
    std::deque<T> items;
    size_t capacity;
    bool closed=false;
public:
    size_t max_depth=0;
    double pop_seconds=0, push_seconds=0;
    explicit Queue(size_t n):capacity(n) { if (!n) throw std::invalid_argument("queue capacity must be positive"); }
    bool push(T item) {
        auto start=Clock::now();
        std::unique_lock<std::mutex> lock(mu);
        writable.wait(lock,[&]{return closed || items.size()<capacity;});
        push_seconds+=seconds(Clock::now()-start);
        if (closed) return false;
        items.push_back(std::move(item));
        max_depth=std::max(max_depth,items.size());
        readable.notify_one();
        return true;
    }
    bool pop(T &item, double *measured=nullptr) {
        auto start=Clock::now();
        std::unique_lock<std::mutex> lock(mu);
        readable.wait(lock,[&]{return closed || !items.empty();});
        double duration=seconds(Clock::now()-start);
        pop_seconds+=duration;
        if (measured) *measured=duration;
        if (items.empty()) return false;
        item=std::move(items.front()); items.pop_front();
        writable.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mu); closed=true;
        readable.notify_all(); writable.notify_all();
    }
};

struct Data { std::vector<Input> x; std::vector<int> y; };
uint32_t u32(std::istream &s) {
    unsigned char b[4]; s.read(reinterpret_cast<char*>(b),4);
    if (!s) throw std::runtime_error("truncated IDX header");
    return (uint32_t(b[0])<<24)|(uint32_t(b[1])<<16)|(uint32_t(b[2])<<8)|b[3];
}
Data load(const std::string &root,const std::string &prefix,size_t limit) {
    std::ifstream images(root+"/"+prefix+"-images-idx3-ubyte",std::ios::binary);
    std::ifstream labels(root+"/"+prefix+"-labels-idx1-ubyte",std::ios::binary);
    if(!images || !labels) throw std::runtime_error("MNIST IDX files missing; run scripts/download_mnist.py");
    uint32_t im=u32(images), n=u32(images), rows=u32(images), cols=u32(images);
    uint32_t lm=u32(labels), ln=u32(labels);
    if(im!=2051 || lm!=2049 || rows!=28 || cols!=28 || n!=ln || limit>n)
        throw std::runtime_error("invalid IDX dimensions, counts, or requested subset");
    Data d; d.x.resize(limit); d.y.resize(limit);
    for(size_t i=0;i<limit;++i) {
        unsigned char pixels[D], label;
        images.read(reinterpret_cast<char*>(pixels),D); labels.read(reinterpret_cast<char*>(&label),1);
        if(!images || !labels || label>=C) throw std::runtime_error("invalid/truncated IDX data");
        for(int j=0;j<D;++j) d.x[i][j]=pixels[j]/255.0;
        d.y[i]=label;
    }
    return d;
}
struct Model {
    std::array<double,D*H> w0{};
    Hidden b0{};
    std::array<double,H*C> w1{};
    Logits b1{};
    explicit Model(unsigned seed) {
        std::mt19937 r0(seed),r1(seed+1);
        std::normal_distribution<double> n0(0,std::sqrt(2.0/D)),n1(0,std::sqrt(1.0/H));
        for(auto &w:w0) w=n0(r0);
        for(auto &w:w1) w=n1(r1);
    }
    Hidden forward0(const Input &x) const {
        Hidden h{};
        for(int j=0;j<H;++j) {
            double a=b0[j];
            for(int i=0;i<D;++i) a+=w0[j*D+i]*x[i];
            h[j]=std::max(0.0,a);
        }
        return h;
    }
    Logits forward1(const Hidden &h) const {
        Logits z{};
        for(int k=0;k<C;++k) {
            z[k]=b1[k];
            for(int j=0;j<H;++j) z[k]+=w1[k*H+j]*h[j];
        }
        return z;
    }
    static double loss(const Logits &z,int label,Logits *delta=nullptr) {
        double maximum=*std::max_element(z.begin(),z.end()), sum=0;
        Logits p{};
        for(int k=0;k<C;++k) {p[k]=std::exp(z[k]-maximum);sum+=p[k];}
        if(delta) {
            for(int k=0;k<C;++k) (*delta)[k]=p[k]/sum;
            (*delta)[label]-=1;
        }
        return maximum+std::log(sum)-z[label];
    }
    // Compute dL/dh using pre-update W1, then update only the output layer.
    double update1(const Hidden &h,int label,double lr,Hidden &dh) {
        Logits delta{};double l=loss(forward1(h),label,&delta);
        dh.fill(0);
        for(int k=0;k<C;++k) for(int j=0;j<H;++j) dh[j]+=w1[k*H+j]*delta[k];
        for(int k=0;k<C;++k) {
            for(int j=0;j<H;++j) w1[k*H+j]-=lr*delta[k]*h[j];
            b1[k]-=lr*delta[k];
        }
        return l;
    }
    // h is the cached FORWARD activation, so its ReLU mask belongs to origin.
    void update0(const Input &x,const Hidden &h,const Hidden &dh,double lr) {
        for(int j=0;j<H;++j) if(h[j]>0) {
            for(int i=0;i<D;++i) w0[j*D+i]-=lr*dh[j]*x[i];
            b0[j]-=lr*dh[j];
        }
    }
    double checksum() const {
        double s=0;size_t i=1;
        for(const auto &v:w0) s+=v*double(i++);
        for(const auto &v:w1) s+=v*double(i++);
        for(const auto &v:b0) s+=v*double(i++);
        for(const auto &v:b1) s+=v*double(i++);
        return s;
    }
};
std::pair<double,double> evaluate(const Model &m,const Data &d) {
    double l=0;size_t correct=0;
    for(size_t i=0;i<d.x.size();++i) {
        auto z=m.forward1(m.forward0(d.x[i])); l+=Model::loss(z,d.y[i]);
        correct+=std::distance(z.begin(),std::max_element(z.begin(),z.end()))==d.y[i];
    }
    return {l/d.x.size(),100.0*correct/d.x.size()};
}
struct Train { size_t id,index; };
struct Activation { size_t id;int label;uint64_t origin;Hidden h; };
struct Gradient { size_t id;uint64_t origin;Hidden dh;double loss; };
using Work=std::variant<Train,Gradient>;
struct Result { size_t id;uint64_t age;bool accepted;double loss; };
struct WorkerStats { double active=0, pop_wait=0;size_t processed=0; };
struct Config {
    std::string mode="sequential",data="data/mnist";
    int epochs=5,tau=4,window=8,train=5000,test=1000;unsigned seed=12345;double lr=.01;
};
bool accept(uint64_t version,uint64_t origin,int tau) {
    if(origin>version) throw std::logic_error("gradient originates in a future version");
    return version-origin<=static_cast<uint64_t>(tau);
}
struct EpochStats {
    size_t accepted=0,discarded=0,output=0,max_age=0,max_accepted_age=0,max_pending=0;
    double age_sum=0,accepted_age_sum=0,loss_sum=0;
};
void add(EpochStats &s,const Result &r) {
    s.loss_sum+=r.loss;s.age_sum+=r.age;s.max_age=std::max(s.max_age,size_t(r.age));s.output++;
    if(r.accepted) {s.accepted++;s.accepted_age_sum+=r.age;s.max_accepted_age=std::max(s.max_accepted_age,size_t(r.age));}
    else s.discarded++;
}
void train(const Config &c) {
    Data data=load(c.data,"train",c.train),test=load(c.data,"t10k",c.test);
    Model model(c.seed);
    std::mt19937 shuffle(c.seed);std::vector<size_t> order(data.x.size());std::iota(order.begin(),order.end(),0);
    size_t capacity=2*size_t(c.window);
    Queue<Work> inbox(capacity);Queue<Activation> activations(capacity);Queue<Result> results(capacity);
    WorkerStats w0,w1;
    std::thread hidden,output;
    if(c.mode=="workers") {
        hidden=std::thread([&]{
            struct Cache {size_t index;Hidden h;};std::map<size_t,Cache> pending;
            uint64_t version=0;Work msg;double wait=0;
            while(inbox.pop(msg,&wait)) {
                w0.pop_wait+=wait;auto start=Clock::now();
                if(auto *t=std::get_if<Train>(&msg)) {
                    Hidden h=model.forward0(data.x[t->index]);
                    pending.emplace(t->id,Cache{t->index,h});
                    activations.push({t->id,data.y[t->index],version,h});
                } else {
                    const auto &g=std::get<Gradient>(msg);auto it=pending.find(g.id);
                    if(it==pending.end()) throw std::logic_error("gradient cache miss");
                    uint64_t age=version-g.origin;bool ok=accept(version,g.origin,c.tau);
                    if(ok) {model.update0(data.x[it->second.index],it->second.h,g.dh,c.lr);version++;}
                    pending.erase(it);results.push({g.id,age,ok,g.loss});w0.processed++;
                }
                w0.active+=seconds(Clock::now()-start);
            }
            w0.pop_wait+=wait;activations.close();
        });
        output=std::thread([&]{
            Activation a;double wait=0;
            while(activations.pop(a,&wait)) {
                w1.pop_wait+=wait;auto start=Clock::now();Hidden dh{};
                double l=model.update1(a.h,a.label,c.lr,dh);
                inbox.push(Gradient{a.id,a.origin,dh,l});w1.processed++;
                w1.active+=seconds(Clock::now()-start);
            }
            w1.pop_wait+=wait;
        });
    }
    auto initial=evaluate(model,test); // Workers have no tasks during this evaluation.
    std::cerr<<std::setprecision(12)<<"initial_test_loss="<<initial.first<<" initial_accuracy="<<initial.second<<'\n';
    std::cout<<"epoch,training_loss,test_loss,test_accuracy_percent,elapsed_training_seconds,epoch_training_seconds,accepted,discarded,output_updates,mean_age,mean_accepted_age,max_age,max_accepted_age,max_pending,parameter_checksum\n";
    double total=0;size_t id=0;
    for(int e=1;e<=c.epochs;++e) {
        std::shuffle(order.begin(),order.end(),shuffle);EpochStats s;auto start=Clock::now();
        if(c.mode=="sequential") {
            for(size_t i:order) {
                Hidden h=model.forward0(data.x[i]),dh{};double l=model.update1(h,data.y[i],c.lr,dh);
                model.update0(data.x[i],h,dh,c.lr);add(s,{id++,0,true,l});
            }
            s.max_pending=1;
        } else {
            size_t submitted=0,completed=0;
            while(completed<order.size()) {
                while(submitted<order.size() && submitted-completed<size_t(c.window)) {
                    inbox.push(Train{id++,order[submitted++]});s.max_pending=std::max(s.max_pending,submitted-completed);
                }
                Result r;if(!results.pop(r)) throw std::runtime_error("unexpected result-queue closure");
                add(s,r);completed++;
            }
        }
        double epoch_time=seconds(Clock::now()-start);total+=epoch_time;
        // Every output update precedes its gradient/result. Draining results
        // establishes queue mutex happens-before edges for both layer writes.
        auto metric=evaluate(model,test);
        std::cout<<std::fixed<<std::setprecision(12)<<e<<','<<s.loss_sum/c.train<<','<<metric.first<<','<<metric.second<<','<<total<<','<<epoch_time<<','<<s.accepted<<','<<s.discarded<<','<<s.output<<','<<s.age_sum/c.train<<','<<(s.accepted?s.accepted_age_sum/s.accepted:0)<<','<<s.max_age<<','<<s.max_accepted_age<<','<<s.max_pending<<','<<model.checksum()<<'\n';
    }
    if(c.mode=="workers") {inbox.close();hidden.join();output.join();}
    std::cerr<<"queue_capacity="<<capacity<<" hidden_inbox_max="<<inbox.max_depth<<" activation_queue_max="<<activations.max_depth<<" result_queue_max="<<results.max_depth<<'\n';
    std::cerr<<"w0_active_seconds="<<w0.active<<" w0_pop_seconds="<<w0.pop_wait<<" w1_active_seconds="<<w1.active<<" w1_pop_seconds="<<w1.pop_wait<<'\n';
    std::cerr<<"inbox_push_seconds="<<inbox.push_seconds<<" activation_push_seconds="<<activations.push_seconds<<" result_push_seconds="<<results.push_seconds<<'\n';
    std::cerr<<"Worker pop durations include evaluation/startup/drain intervals. They are not CPU utilization.\n";
}
void require(bool b,const std::string &message) {if(!b) throw std::runtime_error(message);}
void self_test() {
    require(accept(105,101,4) && !accept(105,101,2) && accept(101,101,0),"age policy");
    bool future=false;try{accept(1,2,4);}catch(const std::logic_error&){future=true;}require(future,"future version rejection");
    Queue<int> q(1);require(q.push(5),"push");q.close();int i=0;require(q.pop(i)&&i==5&&!q.pop(i)&&!q.push(2),"close and drain");
    Queue<int> empty(1);std::promise<void> ready;auto waiting=std::async(std::launch::async,[&]{ready.set_value();int x;return empty.pop(x);});
    ready.get_future().wait();empty.close();require(!waiting.get(),"close wakes consumer");
    Queue<int> full(1);full.push(1);std::promise<void> ready2;auto blocked=std::async(std::launch::async,[&]{ready2.set_value();return full.push(2);});
    ready2.get_future().wait();full.close();require(!blocked.get(),"close wakes producer");
    Model m(12345);Input x{};for(int j=0;j<D;++j)x[j]=(j%17)/17.0;
    auto h=m.forward0(x);Model updated=m;Hidden dh{};double lr=1e-4;updated.update1(h,3,lr,dh);updated.update0(x,h,dh,lr);
    auto objective=[&]{return Model::loss(m.forward1(m.forward0(x)),3);};
    auto check=[&](double &p,double old,double after){double eps=1e-6;p=old+eps;double plus=objective();p=old-eps;double minus=objective();p=old;double numerical=(plus-minus)/(2*eps),analytic=(old-after)/lr;require(std::abs(numerical-analytic)<1e-5*(1+std::abs(numerical)),"finite-difference gradient mismatch");};
    for(int j=0;j<H;j+=3) for(int k=0;k<D;k+=101)check(m.w0[j*D+k],m.w0[j*D+k],updated.w0[j*D+k]);
    for(int k=0;k<C;++k) {check(m.b1[k],m.b1[k],updated.b1[k]);for(int j=0;j<H;j+=5)check(m.w1[k*H+j],m.w1[k*H+j],updated.w1[k*H+j]);}
    for(int j=0;j<H;++j)check(m.b0[j],m.b0[j],updated.b0[j]);
    Logits huge{};huge.fill(10000);require(std::isfinite(Model::loss(huge,0)),"stable softmax");
    std::cout<<"PASS finite-difference gradients, stable loss, version policy, queue drain, close/wakeup\n";
}
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--self-test") {self_test();return 0;}
        Config c;
        for(int i=1;i<argc;++i) {
            std::string k=argv[i];
            if(k=="--help") {std::cout<<"mlp --mode sequential|workers --tau N --in-flight N --epochs N --train N --test N --seed N --lr F --data PATH\n";return 0;}
            if(i+1>=argc) throw std::invalid_argument("missing argument for "+k);
            std::string v=argv[++i];
            if(k=="--mode")c.mode=v;else if(k=="--data")c.data=v;else if(k=="--tau")c.tau=std::stoi(v);else if(k=="--in-flight")c.window=std::stoi(v);else if(k=="--epochs")c.epochs=std::stoi(v);else if(k=="--train")c.train=std::stoi(v);else if(k=="--test")c.test=std::stoi(v);else if(k=="--seed")c.seed=static_cast<unsigned>(std::stoul(v));else if(k=="--lr")c.lr=std::stod(v);else throw std::invalid_argument("unknown flag: "+k);
        }
        if(c.window<=0 || c.epochs<=0 || c.train<=0 || c.test<=0 || c.lr<=0 || !std::isfinite(c.lr) || c.tau<0)throw std::invalid_argument("counts and learning rate must be positive; tau must be nonnegative");
        if(c.mode!="workers" && c.mode!="sequential")throw std::invalid_argument("invalid mode");
        train(c);return 0;
    }catch(const std::exception &e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
}
