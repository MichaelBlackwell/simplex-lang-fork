# Simplex Standard Library Reference

**Version 0.7.0**

The Simplex standard library provides core functionality for I/O, collections, networking, and more. All modules are written in pure Simplex.

---

## Module Overview

| Module | Description |
|--------|-------------|
| `std::io` | I/O traits and buffered streams |
| `std::fs` | File system operations |
| `std::collections` | Data structures |
| `std::string` | String manipulation |
| `std::time` | Time and duration |
| `std::math` | Mathematical functions |
| `std::net` | Networking |
| `std::env` | Environment and process info |
| `std::process` | Process execution |
| `std::fmt` | Formatting and display |
| `simplex_learning` | Real-time learning library (v0.7.0) |

---

## std::io

Input/output traits and buffered streams.

### Traits

```simplex
trait Read {
    fn read(self, buf: &mut [u8]) -> Result<i64, IoError>
    fn read_exact(self, buf: &mut [u8]) -> Result<(), IoError>
    fn read_to_end(self, buf: &mut Vec<u8>) -> Result<i64, IoError>
    fn read_to_string(self, s: &mut String) -> Result<i64, IoError>
}

trait Write {
    fn write(self, buf: &[u8]) -> Result<i64, IoError>
    fn write_all(self, buf: &[u8]) -> Result<(), IoError>
    fn flush(self) -> Result<(), IoError>
}

trait BufRead: Read {
    fn fill_buf(self) -> Result<&[u8], IoError>
    fn consume(self, amt: i64)
    fn read_line(self, buf: &mut String) -> Result<i64, IoError>
    fn lines(self) -> Lines<Self>
}

trait Seek {
    fn seek(self, pos: SeekFrom) -> Result<i64, IoError>
    fn rewind(self) -> Result<(), IoError>
    fn stream_position(self) -> Result<i64, IoError>
}
```

### Types

```simplex
type BufReader<R: Read>     // Buffered reader
type BufWriter<W: Write>    // Buffered writer
type Cursor<T>              // In-memory buffer
type Stdin                  // Standard input
type Stdout                 // Standard output
type Stderr                 // Standard error
```

### Functions

```simplex
fn stdin() -> Stdin
fn stdout() -> Stdout
fn stderr() -> Stderr
fn copy<R: Read, W: Write>(reader: &mut R, writer: &mut W) -> Result<i64, IoError>
fn empty() -> Empty
fn repeat(byte: u8) -> Repeat
fn sink() -> Sink
```

---

## std::fs

File system operations.

### Types

```simplex
type File                   // File handle
type OpenOptions            // File open configuration
type Metadata               // File metadata
type Permissions            // File permissions
type ReadDir                // Directory iterator
type DirEntry               // Directory entry
```

### File Operations

```simplex
// Open files
fn File::open(path: &str) -> Result<File, IoError>
fn File::create(path: &str) -> Result<File, IoError>
fn File::create_new(path: &str) -> Result<File, IoError>

// Read/write convenience
fn read(path: &str) -> Result<Vec<u8>, IoError>
fn read_to_string(path: &str) -> Result<String, IoError>
fn write(path: &str, contents: &[u8]) -> Result<(), IoError>
```

### Directory Operations

```simplex
fn read_dir(path: &str) -> Result<ReadDir, IoError>
fn create_dir(path: &str) -> Result<(), IoError>
fn create_dir_all(path: &str) -> Result<(), IoError>
fn remove_dir(path: &str) -> Result<(), IoError>
fn remove_dir_all(path: &str) -> Result<(), IoError>
```

### Path Operations

```simplex
fn canonicalize(path: &str) -> Result<String, IoError>
fn metadata(path: &str) -> Result<Metadata, IoError>
fn copy(from: &str, to: &str) -> Result<i64, IoError>
fn rename(from: &str, to: &str) -> Result<(), IoError>
fn remove_file(path: &str) -> Result<(), IoError>
fn exists(path: &str) -> bool
fn temp_dir() -> String
```

---

## std::collections

Core data structures.

### Vec<T> - Dynamic Array

```simplex
type Vec<T> {
    fn new() -> Vec<T>
    fn with_capacity(capacity: i64) -> Vec<T>
    fn len(self) -> i64
    fn is_empty(self) -> bool
    fn push(self, value: T)
    fn pop(self) -> Option<T>
    fn insert(self, index: i64, value: T)
    fn remove(self, index: i64) -> T
    fn get(self, index: i64) -> Option<&T>
    fn first(self) -> Option<&T>
    fn last(self) -> Option<&T>
    fn contains(self, value: &T) -> bool
    fn sort(self)
    fn reverse(self)
    fn iter(self) -> VecIter<T>
}
```

### Map<K, V> - Hash Map

```simplex
type Map<K: Hash + Eq, V> {
    fn new() -> Map<K, V>
    fn insert(self, key: K, value: V) -> Option<V>
    fn get(self, key: &K) -> Option<&V>
    fn remove(self, key: &K) -> Option<V>
    fn contains_key(self, key: &K) -> bool
    fn len(self) -> i64
    fn is_empty(self) -> bool
    fn keys(self) -> MapKeys<K, V>
    fn values(self) -> MapValues<K, V>
    fn iter(self) -> MapIter<K, V>
    fn entry(self, key: K) -> Entry<K, V>
}
```

### Set<T> - Hash Set

```simplex
type Set<T: Hash + Eq> {
    fn new() -> Set<T>
    fn insert(self, value: T) -> bool
    fn remove(self, value: &T) -> bool
    fn contains(self, value: &T) -> bool
    fn union(self, other: &Set<T>) -> Set<T>
    fn intersection(self, other: &Set<T>) -> Set<T>
    fn difference(self, other: &Set<T>) -> Set<T>
    fn is_subset(self, other: &Set<T>) -> bool
}
```

### Deque<T> - Double-Ended Queue

```simplex
type Deque<T> {
    fn new() -> Deque<T>
    fn push_front(self, value: T)
    fn push_back(self, value: T)
    fn pop_front(self) -> Option<T>
    fn pop_back(self) -> Option<T>
    fn front(self) -> Option<&T>
    fn back(self) -> Option<&T>
}
```

### PriorityQueue<T> - Binary Heap

```simplex
type PriorityQueue<T: Ord> {
    fn new() -> PriorityQueue<T>
    fn push(self, value: T)
    fn pop(self) -> Option<T>
    fn peek(self) -> Option<&T>
}
```

---

## std::string

String manipulation and UTF-8 handling.

### String Type

```simplex
type String {
    fn new() -> String
    fn from_str(s: &str) -> String
    fn from_utf8(bytes: Vec<u8>) -> Result<String, Utf8Error>
    fn len(self) -> i64
    fn is_empty(self) -> bool
    fn push(self, ch: char)
    fn push_str(self, s: &str)
    fn pop(self) -> Option<char>
    fn chars(self) -> Chars
    fn lines(self) -> Lines
    fn split(self, delimiter: &str) -> Split
    fn trim(self) -> String
    fn to_lowercase(self) -> String
    fn to_uppercase(self) -> String
    fn contains(self, pattern: &str) -> bool
    fn starts_with(self, prefix: &str) -> bool
    fn ends_with(self, suffix: &str) -> bool
    fn find(self, pattern: &str) -> Option<i64>
    fn replace(self, from: &str, to: &str) -> String
    fn repeat(self, n: i64) -> String
    fn parse<T: FromStr>(self) -> Result<T, T::Err>
}
```

### StringBuilder

```simplex
type StringBuilder {
    fn new() -> StringBuilder
    fn append(self, s: &str) -> &mut StringBuilder
    fn append_char(self, ch: char) -> &mut StringBuilder
    fn append_line(self, s: &str) -> &mut StringBuilder
    fn build(self) -> String
}
```

### Character Methods

```simplex
impl char {
    fn is_alphabetic(self) -> bool
    fn is_alphanumeric(self) -> bool
    fn is_numeric(self) -> bool
    fn is_whitespace(self) -> bool
    fn is_uppercase(self) -> bool
    fn is_lowercase(self) -> bool
    fn to_uppercase(self) -> char
    fn to_lowercase(self) -> char
    fn to_digit(self, radix: u32) -> Option<u32>
}
```

---

## std::time

Time and duration handling.

### Duration

```simplex
type Duration {
    fn zero() -> Duration
    fn from_secs(secs: i64) -> Duration
    fn from_millis(millis: i64) -> Duration
    fn from_micros(micros: i64) -> Duration
    fn from_nanos(nanos: i64) -> Duration
    fn as_secs(self) -> i64
    fn as_millis(self) -> i64
    fn as_secs_f64(self) -> f64
    fn is_zero(self) -> bool
}
```

### Instant (Monotonic Clock)

```simplex
type Instant {
    fn now() -> Instant
    fn elapsed(self) -> Duration
    fn duration_since(self, earlier: Instant) -> Duration
}
```

### SystemTime (Wall Clock)

```simplex
type SystemTime {
    fn now() -> SystemTime
    fn elapsed(self) -> Result<Duration, SystemTimeError>
    fn duration_since(self, earlier: SystemTime) -> Result<Duration, SystemTimeError>
}

const UNIX_EPOCH: SystemTime
```

### Functions

```simplex
fn sleep(dur: Duration)
fn sleep_ms(ms: i64)
fn timestamp() -> i64           // Unix timestamp in seconds
fn timestamp_millis() -> i64    // Unix timestamp in milliseconds
```

### Timer and Timeout

```simplex
type Timer {
    fn new(interval: Duration) -> Timer
    fn tick(self) -> bool
    fn wait(self)
    fn reset(self)
}

type Timeout {
    fn new(duration: Duration) -> Timeout
    fn is_expired(self) -> bool
    fn remaining(self) -> Duration
}
```

---

## std::math

Mathematical functions and constants.

### Constants

```simplex
const PI: f64 = 3.141592653589793
const E: f64 = 2.718281828459045
const TAU: f64 = 6.283185307179586
const SQRT_2: f64 = 1.4142135623730951
const LN_2: f64 = 0.6931471805599453
const LN_10: f64 = 2.302585092994046
```

### Basic Functions

```simplex
fn min<T: Ord>(a: T, b: T) -> T
fn max<T: Ord>(a: T, b: T) -> T
fn clamp<T: Ord>(value: T, min: T, max: T) -> T
fn abs(x: i64) -> i64
fn abs_f64(x: f64) -> f64
fn gcd(a: i64, b: i64) -> i64
fn lcm(a: i64, b: i64) -> i64
```

### Exponential and Logarithmic

```simplex
fn sqrt(x: f64) -> f64
fn cbrt(x: f64) -> f64
fn pow(base: f64, exp: f64) -> f64
fn exp(x: f64) -> f64
fn ln(x: f64) -> f64
fn log(x: f64, base: f64) -> f64
fn log2(x: f64) -> f64
fn log10(x: f64) -> f64
```

### Trigonometric

```simplex
fn sin(x: f64) -> f64
fn cos(x: f64) -> f64
fn tan(x: f64) -> f64
fn asin(x: f64) -> f64
fn acos(x: f64) -> f64
fn atan(x: f64) -> f64
fn atan2(y: f64, x: f64) -> f64
```

### Hyperbolic

```simplex
fn sinh(x: f64) -> f64
fn cosh(x: f64) -> f64
fn tanh(x: f64) -> f64
fn asinh(x: f64) -> f64
fn acosh(x: f64) -> f64
fn atanh(x: f64) -> f64
```

### Rounding

```simplex
fn floor(x: f64) -> f64
fn ceil(x: f64) -> f64
fn round(x: f64) -> f64
fn trunc(x: f64) -> f64
fn fract(x: f64) -> f64
```

### Special Functions

```simplex
fn gamma(x: f64) -> f64
fn lgamma(x: f64) -> f64
fn factorial(n: i64) -> i64
fn binomial(n: i64, k: i64) -> i64
fn erf(x: f64) -> f64
fn erfc(x: f64) -> f64
```

### Interpolation

```simplex
fn lerp(a: f64, b: f64, t: f64) -> f64
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64
fn degrees(radians: f64) -> f64
fn radians(degrees: f64) -> f64
```

---

## std::net

Networking primitives.

### IP Addresses

```simplex
type Ipv4Addr {
    fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr
    fn localhost() -> Ipv4Addr
    fn unspecified() -> Ipv4Addr
    fn is_loopback(self) -> bool
    fn is_private(self) -> bool
}

type Ipv6Addr {
    fn new(a: u16, ..., h: u16) -> Ipv6Addr
    fn localhost() -> Ipv6Addr
}

enum IpAddr { V4(Ipv4Addr), V6(Ipv6Addr) }
```

### Socket Addresses

```simplex
type SocketAddr {
    fn new(ip: IpAddr, port: u16) -> SocketAddr
    fn ip(self) -> IpAddr
    fn port(self) -> u16
    fn parse(s: &str) -> Result<SocketAddr, ParseAddrError>
}
```

### TCP

```simplex
type TcpListener {
    fn bind(addr: &str) -> Result<TcpListener, IoError>
    fn accept(self) -> Result<(TcpStream, SocketAddr), IoError>
    fn incoming(self) -> Incoming
    fn set_nonblocking(self, nonblocking: bool) -> Result<(), IoError>
}

type TcpStream {
    fn connect(addr: &str) -> Result<TcpStream, IoError>
    fn peer_addr(self) -> Result<SocketAddr, IoError>
    fn shutdown(self, how: Shutdown) -> Result<(), IoError>
    fn set_nodelay(self, nodelay: bool) -> Result<(), IoError>
}

// TcpStream implements Read and Write
```

### UDP

```simplex
type UdpSocket {
    fn bind(addr: &str) -> Result<UdpSocket, IoError>
    fn recv_from(self, buf: &mut [u8]) -> Result<(i64, SocketAddr), IoError>
    fn send_to(self, buf: &[u8], addr: &str) -> Result<i64, IoError>
    fn connect(self, addr: &str) -> Result<(), IoError>
    fn recv(self, buf: &mut [u8]) -> Result<i64, IoError>
    fn send(self, buf: &[u8]) -> Result<i64, IoError>
}
```

### DNS

```simplex
fn lookup_host(host: &str) -> Result<Vec<IpAddr>, IoError>
fn resolve(host: &str, port: u16) -> Result<Vec<SocketAddr>, IoError>
```

---

## std::env

Environment and process information.

### Command Line Arguments

```simplex
fn args() -> Args               // Iterator over arguments
fn args_os() -> Vec<String>     // All arguments as vector
```

### Environment Variables

```simplex
fn var(key: &str) -> Result<String, VarError>
fn var_os(key: &str) -> Option<String>
fn set_var(key: &str, value: &str)
fn remove_var(key: &str)
fn vars() -> Vars              // Iterator over all variables
```

### Directories

```simplex
fn current_dir() -> Result<String, IoError>
fn set_current_dir(path: &str) -> Result<(), IoError>
fn home_dir() -> Option<String>
fn temp_dir() -> String
fn current_exe() -> Result<String, IoError>
```

### Platform Information

```simplex
fn os() -> &'static str        // "linux", "macos", "windows"
fn arch() -> &'static str      // "x86_64", "aarch64"
fn family() -> &'static str    // "unix", "windows"
fn num_cpus() -> i64
fn is_linux() -> bool
fn is_macos() -> bool
fn is_windows() -> bool
fn is_unix() -> bool
```

### User Information

```simplex
fn username() -> Option<String>
fn uid() -> i64
fn gid() -> i64
fn is_root() -> bool
```

### Process Control

```simplex
fn pid() -> i64
fn ppid() -> i64
fn exit(code: i64) -> !
fn abort() -> !
```

---

## std::process

Process execution.

### Command Builder

```simplex
type Command {
    fn new(program: &str) -> Command
    fn arg(self, arg: &str) -> &mut Command
    fn args<I>(self, args: I) -> &mut Command
    fn env(self, key: &str, value: &str) -> &mut Command
    fn current_dir(self, dir: &str) -> &mut Command
    fn stdin(self, cfg: Stdio) -> &mut Command
    fn stdout(self, cfg: Stdio) -> &mut Command
    fn stderr(self, cfg: Stdio) -> &mut Command
    fn spawn(self) -> Result<Child, IoError>
    fn output(self) -> Result<Output, IoError>
    fn status(self) -> Result<ExitStatus, IoError>
}
```

### Child Process

```simplex
type Child {
    fn id(self) -> i64
    fn kill(self) -> Result<(), IoError>
    fn wait(self) -> Result<ExitStatus, IoError>
    fn try_wait(self) -> Result<Option<ExitStatus>, IoError>
}
```

### Types

```simplex
enum Stdio { Inherit, Piped, Null }

type ExitStatus {
    fn success(self) -> bool
    fn code(self) -> Option<i64>
}

type Output {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}
```

### Convenience Functions

```simplex
fn run(program: &str, args: &[&str]) -> Result<String, IoError>
fn run_checked(program: &str, args: &[&str]) -> Result<(), IoError>
fn which(program: &str) -> Option<String>
```

---

## std::fmt

Formatting and display.

### Traits

```simplex
trait Display {
    fn fmt(self, f: &mut Formatter) -> Result<(), FormatError>
}

trait Debug {
    fn fmt(self, f: &mut Formatter) -> Result<(), FormatError>
}

trait Binary { ... }
trait Octal { ... }
trait LowerHex { ... }
trait UpperHex { ... }
```

### Formatter

```simplex
type Formatter {
    fn write_str(self, s: &str) -> Result<(), FormatError>
    fn write_char(self, c: char) -> Result<(), FormatError>
    fn width(self) -> Option<i64>
    fn precision(self) -> Option<i64>
    fn fill(self) -> char
    fn align(self) -> Alignment
    fn alternate(self) -> bool
    fn pad(self, s: &str) -> Result<(), FormatError>
}
```

### Functions

```simplex
fn format(fmt: &str, values: Vec<FormatValue>) -> String
fn write<W: Write>(writer: &mut W, fmt: &str, values: Vec<FormatValue>) -> Result<(), FormatError>
```

---

## Message Passing (std::runtime::channel)

Channel-based communication between actors.

### MPSC Channels

```simplex
fn mpsc_channel<T>(capacity: i64) -> (Sender<T>, Receiver<T>)
fn mpsc_unbounded<T>() -> (Sender<T>, Receiver<T>)

impl Sender<T> {
    fn send(self, value: T) -> Result<(), ChannelError>
    fn try_send(self, value: T) -> Result<(), ChannelError>
    fn is_closed(self) -> bool
}

impl Receiver<T> {
    fn receive(self) -> Result<T, ChannelError>
    fn try_receive(self) -> Option<T>
}
```

### Oneshot Channels

```simplex
fn oneshot<T>() -> (OneshotSender<T>, OneshotReceiver<T>)
```

### Broadcast Channels

```simplex
fn broadcast<T>(capacity: i64) -> Broadcaster<T>

impl Broadcaster<T> {
    fn send(self, value: T) -> Result<(), ChannelError>
    fn subscribe(self, capacity: i64) -> BroadcastReceiver<T>
}
```

### Watch Channels

```simplex
fn watch<T>(initial: T) -> (WatchSender<T>, WatchReceiver<T>)

impl WatchReceiver<T> {
    fn borrow(self) -> &T
    fn has_changed(self) -> bool
}
```

### Select

```simplex
fn select<T>(receivers: Vec<&Receiver<T>>) -> SelectResult<T>
fn select_timeout<T>(receivers: Vec<&Receiver<T>>, timeout_ms: i64) -> SelectResult<T>
```

---

## simplex_learning (v0.7.0)

Real-time continuous learning library for AI specialists.

### Tensor Operations

```simplex
use simplex_learning::tensor::{Tensor, ops};

type Tensor {
    fn zeros(shape: &[i64]) -> Tensor
    fn ones(shape: &[i64]) -> Tensor
    fn randn(shape: &[i64]) -> Tensor
    fn from_slice(data: &[f64], shape: &[i64]) -> Tensor
    fn requires_grad_(self) -> Tensor
    fn grad(self) -> Option<Tensor>
    fn backward(self)
    fn shape(self) -> &[i64]
    fn reshape(self, shape: &[i64]) -> Tensor
}

// Operations
fn ops::matmul(a: &Tensor, b: &Tensor) -> Tensor
fn ops::batch_matmul(a: &Tensor, b: &Tensor) -> Tensor
fn ops::add(a: &Tensor, b: &Tensor) -> Tensor
fn ops::mul(a: &Tensor, b: &Tensor) -> Tensor
fn ops::relu(x: &Tensor) -> Tensor
fn ops::sigmoid(x: &Tensor) -> Tensor
fn ops::softmax(x: &Tensor, dim: i64) -> Tensor
fn ops::mse_loss(pred: &Tensor, target: &Tensor) -> Tensor
fn ops::cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor
```

### Optimizers

```simplex
use simplex_learning::optim::{StreamingSGD, StreamingAdam, AdamW};

type StreamingSGD {
    fn new(lr: f64) -> StreamingSGD
    fn momentum(self, m: f64) -> StreamingSGD
    fn weight_decay(self, wd: f64) -> StreamingSGD
    fn max_grad_norm(self, max: f64) -> StreamingSGD
    fn step(self, params: &mut [Tensor])
}

type StreamingAdam {
    fn new(lr: f64) -> StreamingAdam
    fn betas(self, b1: f64, b2: f64) -> StreamingAdam
    fn eps(self, e: f64) -> StreamingAdam
    fn accumulation_steps(self, steps: i64) -> StreamingAdam
    fn step(self, params: &mut [Tensor])
}

type AdamW {
    fn new(lr: f64) -> AdamW
    fn weight_decay(self, wd: f64) -> AdamW
    fn step(self, params: &mut [Tensor])
}

// Gradient clipping
fn clip_grad_norm(params: &mut [Tensor], max_norm: f64) -> f64
fn clip_grad_value(params: &mut [Tensor], max_value: f64)
```

### Safety

```simplex
use simplex_learning::safety::{SafeLearner, SafeFallback, ConstraintManager};

enum SafeFallback<T> {
    fn with_default(value: T) -> SafeFallback<T>
    fn last_good() -> SafeFallback<T>
    fn with_function(f: fn(&Input) -> T) -> SafeFallback<T>
    fn checkpoint(path: &str) -> SafeFallback<T>
    fn skip_update() -> SafeFallback<T>
}

type SafeLearner<T> {
    fn new(learner: OnlineLearner, fallback: SafeFallback<T>) -> SafeLearner<T>
    fn with_validator(self, f: fn(&T) -> bool) -> SafeLearner<T>
    fn max_failures(self, n: i64) -> SafeLearner<T>
    fn try_process(self, input: &Input, f: fn(&Input) -> T) -> Result<T, SafetyError>
}

type ConstraintManager {
    fn new() -> ConstraintManager
    fn add_soft(self, constraint: Constraint) -> ConstraintManager
    fn add_hard(self, constraint: Constraint) -> ConstraintManager
    fn check(self, metrics: &Metrics) -> ConstraintResult
}
```

### Distributed Learning

```simplex
use simplex_learning::distributed::{
    FederatedLearner, KnowledgeDistiller, HiveBeliefManager, HiveLearningCoordinator
};

// Federated Learning
type FederatedLearner {
    fn new(config: FederatedConfig, params: Vec<Tensor>) -> FederatedLearner
    fn submit_update(self, update: NodeUpdate)
    fn global_params(self) -> Vec<Tensor>
    fn round(self) -> i64
}

enum AggregationStrategy {
    FedAvg,
    WeightedAvg,
    PerformanceWeighted,
    Median,
    TrimmedMean,
    AttentionWeighted,
}

// Knowledge Distillation
type KnowledgeDistiller {
    fn new(config: DistillationConfig) -> KnowledgeDistiller
    fn distillation_loss(self, student: &Tensor, teacher: &Tensor, labels: &Tensor) -> Tensor
}

// Belief Resolution
type HiveBeliefManager {
    fn new(strategy: ConflictResolution) -> HiveBeliefManager
    fn submit_belief(self, belief: Belief)
    fn consensus(self, key: &str) -> f64
    fn all_beliefs(self) -> Vec<Belief>
}

enum ConflictResolution {
    HighestConfidence,
    MostRecent,
    MostEvidence,
    EvidenceWeighted,
    BayesianCombination,
    SemanticWeighted,
    MajorityVote,
}

// Hive Coordinator
type HiveLearningCoordinator {
    fn new(config: HiveLearningConfig, params: Vec<Tensor>) -> HiveLearningCoordinator
    fn register_specialist(self, name: &str, params: Vec<Tensor>)
    fn submit_gradients(self, name: &str, grads: &[Tensor])
    fn submit_belief(self, name: &str, belief: Belief)
    fn step(self)
    fn get_specialist_params(self, name: &str) -> Vec<Tensor>
}
```

### Runtime

```simplex
use simplex_learning::runtime::{OnlineLearner, Checkpoint, Metrics};

type OnlineLearner {
    fn new(params: Vec<Tensor>) -> OnlineLearner
    fn optimizer<O: Optimizer>(self, opt: O) -> OnlineLearner
    fn fallback(self, fb: SafeFallback) -> OnlineLearner
    fn constraint(self, c: Constraint) -> OnlineLearner
    fn forward(self, input: &Tensor) -> Tensor
    fn learn(self, feedback: &FeedbackSignal)
    fn metrics(self) -> Metrics
}

type Checkpoint {
    fn save(path: &str, learner: &OnlineLearner) -> Result<(), IoError>
    fn load(path: &str) -> Result<OnlineLearner, IoError>
}

type Metrics {
    loss: f64,
    lr: f64,
    grad_norm: f64,
    step_count: i64,
    samples_per_sec: f64,
}
```

---

*All standard library modules are written in pure Simplex with no external dependencies.*
