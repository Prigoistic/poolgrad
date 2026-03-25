mod autograd;
mod tensor;
mod nn;
mod plannar;
mod mem;

use autograd::graph::Graph;
use tensor::store::TensorStore;
use tensor::tensor::Tensor;
use nn::linear::Linear;
use nn::loss::mse;
use mem::pool::MemoryPool;
use plannar::lifetime::compute_lifetimes;

struct Config {
    use_pool: bool,
}

fn run_training(config: &Config) -> MemoryPool {
    let mut store = TensorStore::new();
    let mut graph = Graph::new();
    let mut pool = MemoryPool::new();
    pool.enabled = config.use_pool;
    let x_id = store.add(Tensor::new(vec![1.0, 2.0], vec![1, 2], true));
    let target_id = store.add(Tensor::new(vec![3.0, 3.0, 3.0], vec![1, 3], false));

    let linear = Linear::new(2, 3, &mut store);

    for epoch in 0..10 {
        // forward
        let out_id = linear.forward(x_id, &mut store, &mut graph);
        let loss_id = mse(out_id, target_id, &mut store, &mut graph, &mut pool);

        // backward
        graph.backward(&mut store, loss_id);

        // update weights
        let lr = 0.01;

        let weight = store.get_mut(linear.weight_id);
        for i in 0..weight.data.len() {
            weight.data[i] -= lr * weight.grad[i];
        }

        let bias = store.get_mut(linear.bias_id);
        for i in 0..bias.data.len() {
            bias.data[i] -= lr * bias.grad[i];
        }

        println!(
            "Epoch {} Loss: {:?}, Output: {:?}",
            epoch,
            store.get(loss_id).data,
            store.get(out_id).data
        );

        // planner + pool: free intermediate grad buffers at their death step
        let lifetimes = compute_lifetimes(&graph);

        if epoch == 9 {
            for (id, lt) in lifetimes.iter() {
                println!("Tensor {}: birth={}, death={}", id, lt.birth, lt.death);
            }
        }

        // Bucket tensors by death step to avoid scanning all lifetimes each step.
        let mut death_buckets: Vec<Vec<usize>> = vec![Vec::new(); graph.nodes.len()];
        for (tensor_id, lifetime) in lifetimes.iter() {
            if lifetime.death < death_buckets.len() {
                let tensor_id = *tensor_id;
                // Only release intermediate tensors (creator set). Persistent tensors like params/inputs keep grads.
                if store.get(tensor_id).creator.is_some() {
                    death_buckets[lifetime.death].push(tensor_id);
                }
            }
        }

        for (step, _node) in graph.nodes.iter().enumerate() {
            for &tensor_id in &death_buckets[step] {
                let tensor = store.get_mut(tensor_id);
                if !tensor.grad.is_empty() {
                    pool.release(std::mem::take(&mut tensor.grad));
                }
            }
        }

        // reset graph + grads (IMPORTANT)
        graph.nodes.clear();
        for t in &mut store.tensors {
            t.zero_grad();
        }
    }

    pool
}

fn print_metrics(label: &str, pool: &MemoryPool) {
    let total = (pool.allocations + pool.reuses) as f32;
    let reduction = if total > 0.0 {
        (pool.reuses as f32 / total) * 100.0
    } else {
        0.0
    };

    let peak_bytes = pool.peak_memory * std::mem::size_of::<f32>();

    println!("=== {} ===", label);
    println!(
        "Allocations: {}, Reuses: {}, Reduction: {:.2}%",
        pool.allocations, pool.reuses, reduction
    );
    println!("Peak memory: {} bytes", peak_bytes);
}

fn main() {
    let baseline = Config { use_pool: false };
    let pooled = Config { use_pool: true };

    let baseline_pool = run_training(&baseline);
    let pooled_pool = run_training(&pooled);

    print_metrics("Baseline (pool OFF)", &baseline_pool);
    print_metrics("Pooled (pool ON)", &pooled_pool);
}