extern crate ocl;
extern crate nvml_wrapper as nvml;
extern crate openssl;

use ocl::{ProQue, Buffer, MemFlags, Platform, Device, Context};
use nvml::Nvml;
use nvml::enum_wrappers::device::TemperatureSensor;
use openssl::bn::BigNum;
use std::{thread, time::Duration, sync::{Arc, Mutex}};

const KERNEL_SRC: &str = r#"
    int is_prime(ulong n) {
        if (n <= 1) return 0;
        if (n <= 3) return 1;
        if (n % 2 == 0 || n % 3 == 0) return 0;
        for (ulong i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return 0;
        }
        return 1;
    }

    __kernel void test_prime_candidates(__global ulong* candidates, __global int* results, ulong num_candidates) {
        ulong tid = get_global_id(0);
        if (tid < num_candidates) {
            results[tid] = is_prime(candidates[tid]);
        }
    }
"#;

fn generate_large_prime(bits: u32) -> BigNum {
    let mut prime = BigNum::new().unwrap();
    prime.generate_prime(bits as i32, false, None, None).unwrap();
    prime
}

fn main() {
    // Initialize NVML for GPU monitoring
    let nvml = Arc::new(Nvml::init().expect("Failed to initialize NVML"));

    // List available platforms and devices
    let platforms = Platform::list();
    println!("Available platforms:");
    for platform in &platforms {
        println!("Platform: {}", platform.name().unwrap());
        let devices = Device::list_all(&*platform).unwrap();
        for device in &devices {
            println!("  Device: {}", device.name().unwrap());
        }
    }

    // Create ProQue for each device
    let mut pro_ques: Vec<Arc<ProQue>> = vec![];
    for platform in &platforms {
        let devices = Device::list_all(&*platform).unwrap();
        for device in devices {
            let max_threads = 64; // Further reduce the number of threads to avoid resource exhaustion

            // Create a context for the specific platform and device
            let context = Context::builder()
                .platform(*platform)
                .devices(device.clone())
                .build()
                .expect("Failed to create context");

            let pro_que = ProQue::builder()
                .context(context)
                .src(KERNEL_SRC)
                .dims(max_threads) // Use the further reduced number of threads
                .device(device)
                .build()
                .expect("Failed to create ProQue");
            pro_ques.push(Arc::new(pro_que));
        }
    }

    let num_candidates = 100; // Number of prime candidates to generate and test
    let prime_candidates: Vec<u64> = (0..num_candidates)
        .map(|_| generate_large_prime(64).to_dec_str().unwrap().parse::<u64>().unwrap())
        .collect(); // Generate 64-bit prime candidates
    let prime_candidates = Arc::new(prime_candidates);

    let candidates_buffers: Vec<_> = pro_ques.iter().map(|pq| {
        Arc::new(Buffer::<u64>::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(num_candidates as usize) // Use usize instead of u64
            .copy_host_slice(&prime_candidates)
            .build()
            .expect("Failed to create candidates buffer"))
    }).collect();

    let results_buffers: Vec<_> = pro_ques.iter().map(|pq| {
        Arc::new(Buffer::<i32>::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(num_candidates as usize) // Use usize instead of u64
            .build()
            .expect("Failed to create results buffer"))
    }).collect();

    let kernels: Vec<_> = pro_ques.iter().zip(candidates_buffers.iter()).zip(results_buffers.iter()).map(|((pq, cb), rb)| {
        pq.kernel_builder("test_prime_candidates")
            .arg(&**cb) // Dereference Arc
            .arg(&**rb) // Dereference Arc
            .arg(num_candidates as u64)
            .build()
            .expect("Failed to create kernel")
    }).collect();

    // Execute the kernels
    println!("Starting computation...");

    for kernel in &kernels {
        unsafe {
            kernel.enq().expect("Failed to execute kernel");
        }
    }

    let prime_found = Arc::new(Mutex::new(false));

    // Periodically read the status buffer to monitor thread status and GPU utilization
    let mut threads = vec![];
    for (i, (_pq, result_buffer)) in pro_ques.iter().zip(results_buffers.iter()).enumerate() {
        let prime_found = Arc::clone(&prime_found);
        let nvml = Arc::clone(&nvml);
        let result_buffer: Arc<Buffer<i32>> = Arc::clone(result_buffer); // Explicitly specify the type
        let prime_candidates = Arc::clone(&prime_candidates); // Clone the Arc for each thread

        threads.push(thread::spawn(move || {
            let sleep_duration = Duration::from_secs(1);
            let mut elapsed_time = 0;
            let mut results = vec![0i32; num_candidates as usize]; // Number of threads

            loop {
                if *prime_found.lock().unwrap() {
                    break;
                }

                // Read the results buffer
                match result_buffer.read(&mut results).enq() {
                    Ok(_) => {
                        // Print the results of a few threads
                        println!("Prime test results for GPU {}:", i);
                        for j in 0..10.min(results.len()) {
                            println!("  Candidate {}: {}", j, results[j]);
                        }

                        // Check if a prime was found
                        for (j, &res) in results.iter().enumerate() {
                            if res == 1 {
                                println!("Prime found by GPU {}: {}", i, prime_candidates[j]);
                                *prime_found.lock().unwrap() = true;
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to read result buffer: {:?}", e);
                        break;
                    }
                }

                // Monitor GPU utilization and temperature every 10 seconds
                if elapsed_time % 10 == 0 {
                    let device = nvml.device_by_index(i as u32).expect("Failed to get device");
                    let utilization = device.utilization_rates().expect("Failed to get utilization rates");
                    let temperature = device.temperature(TemperatureSensor::Gpu).expect("Failed to get temperature");

                    println!("GPU {}: Utilization: {}%, Temperature: {}°C", i, utilization.gpu, temperature);
                }

                thread::sleep(sleep_duration);
                elapsed_time += 1;
            }
        }));
    }

    for t in threads {
        t.join().unwrap();
    }

    if !*prime_found.lock().unwrap() {
        println!("No prime found in the range.");
    }

    println!("Computation finished.");
}
