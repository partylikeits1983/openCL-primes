extern crate ocl;
extern crate nvml_wrapper as nvml;

use ocl::{ProQue, Buffer, MemFlags, Platform, Device, Context};
use nvml::Nvml;
use nvml::enum_wrappers::device::TemperatureSensor;
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

    __kernel void search_for_large_prime(__global ulong* result, __global ulong* status, ulong start, ulong end) {
        ulong tid = get_global_id(0);
        ulong step = get_global_size(0);  // Number of threads
        for (ulong i = start + tid; i <= end; i += step) {
            status[tid] = i; // Write the current number being tested to the status buffer
            if (is_prime(i)) {
                result[0] = i;
                return;
            }
        }
    }
"#;

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

    // Define the range to search for primes
    let start = 10_000_000_000_000u64;
    let end = start + 1_000_000_000u64; // Adjust this as needed for a larger workload

    // Create ProQue for each device
    let mut pro_ques: Vec<Arc<ProQue>> = vec![];
    for platform in &platforms {
        let devices = Device::list_all(&*platform).unwrap();
        for device in devices {
            let max_threads = 1024; // Limiting to 1024 threads to reduce resource usage

            // Create a context for the specific platform and device
            let context = Context::builder()
                .platform(*platform)
                .devices(device.clone())
                .build()
                .expect("Failed to create context");

            let pro_que = ProQue::builder()
                .context(context)
                .src(KERNEL_SRC)
                .dims(max_threads) // Use the limited number of threads
                .device(device)
                .build()
                .expect("Failed to create ProQue");
            pro_ques.push(Arc::new(pro_que));
        }
    }

    let result_buffers: Vec<_> = pro_ques.iter().map(|pq| {
        Arc::new(Buffer::<u64>::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(1)
            .build()
            .expect("Failed to create result buffer"))
    }).collect();

    let status_buffers: Vec<_> = pro_ques.iter().map(|pq| {
        Arc::new(Buffer::<u64>::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(pq.dims().to_len())
            .build()
            .expect("Failed to create status buffer"))
    }).collect();

    let result_buffers = Arc::new(result_buffers);
    let status_buffers = Arc::new(status_buffers);

    let kernels: Vec<_> = pro_ques.iter().zip(result_buffers.iter()).zip(status_buffers.iter()).map(|((pq, rb), sb)| {
        pq.kernel_builder("search_for_large_prime")
            .arg(&**rb) // Dereference Arc
            .arg(&**sb) // Dereference Arc
            .arg(start)
            .arg(end)
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
    for (i, ((pq, status_buffer), result_buffer)) in pro_ques.iter().zip(status_buffers.iter()).zip(result_buffers.iter()).enumerate() {
        let prime_found = Arc::clone(&prime_found);
        let nvml = Arc::clone(&nvml);
        let status_buffer = Arc::clone(status_buffer);
        let result_buffer = Arc::clone(result_buffer);
        let pq = Arc::clone(pq); // Clone Arc<ProQue> for this thread

        threads.push(thread::spawn(move || {
            let sleep_duration = Duration::from_secs(1);
            let mut elapsed_time = 0;
            let mut status = vec![0u64; pq.dims().to_len()]; // Number of threads

            loop {
                if *prime_found.lock().unwrap() {
                    break;
                }

                // Read the status buffer
                match status_buffer.read(&mut status).enq() {
                    Ok(_) => {
                        // Print the status of a few threads
                        println!("Thread statuses for GPU {}:", i);
                        for j in 0..10.min(status.len()) {
                            println!("  Thread {}: {}", j, status[j]);
                        }

                        // Check if a prime was found
                        let mut result = vec![0u64; 1];
                        match result_buffer.read(&mut result).enq() {
                            Ok(_) => {
                                if result[0] != 0 {
                                    println!("Prime found by GPU {}: {}", i, result[0]);
                                    *prime_found.lock().unwrap() = true;
                                    break;
                                }
                            }
                            Err(e) => {
                                println!("Failed to read result buffer: {:?}", e);
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to read status buffer: {:?}", e);
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

