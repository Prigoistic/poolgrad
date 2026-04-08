mod autograd;
mod config;
mod kernels;
mod memory;
mod nn;
mod planner;
mod runtime;
mod tensor;
fn main() {
    runtime::run();
}
