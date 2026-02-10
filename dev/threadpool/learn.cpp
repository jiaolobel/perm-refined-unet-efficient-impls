// #include <condition_variable>
// #include <functional>
// #include <iostream>
// #include <mutex>
// #include <queue>
// #include <thread>
// #include <vector>

// class ThreadPool {
//   public:
//     ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
//         for (size_t i = 0; i < num_threads; ++i) {
//             threads_.emplace_back([this] {
//                 while (true) {
//                     std::function<void()> task;
//                     {
//                         std::unique_lock<std::mutex> lock(queue_mutex_);
//                         cv_.wait(lock,
//                                  [this] { return !tasks_.empty() || stop_;
//                                  });
//                         if (stop_ && tasks_.empty())
//                             return;
//                         task = std::move(tasks_.front());
//                         tasks_.pop();
//                     }
//                     task();
//                 }
//             });
//         }
//     }

//     ~ThreadPool() {
//         {
//             std::unique_lock<std::mutex> lock(queue_mutex_);
//             stop_ = true;
//         }
//         cv_.notify_all();
//         for (auto &thread : threads_) {
//             thread.join();
//         }
//     }

//     void enqueue(std::function<void()> task) {
//         {
//             std::unique_lock<std::mutex> lock(queue_mutex_);
//             tasks_.emplace(std::move(task));
//         }
//         cv_.notify_one();
//     }

//   private:
//     std::vector<std::thread> threads_;
//     std::queue<std::function<void()>> tasks_;
//     std::mutex queue_mutex_;
//     std::condition_variable cv_;
//     bool stop_ = false;
// };

// int main() {
//     ThreadPool pool(4);
//     for (int i = 0; i < 5; ++i) {
//         pool.enqueue([i] {
//             std::cout << "Task " << i << " is running on thread "
//                       << std::this_thread::get_id() << std::endl;
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//         });
//     }

//     return 0;
// }

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include <chrono>
#include <iostream>

class ThreadPool {
  public:
    ThreadPool(size_t);
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

  private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; i++) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task]() { (*task)(); });
    }

    condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

int main() {
    ThreadPool pool(4);
    std::vector<std::future<int>> results;

    for (int i = 0; i < 8; i++) {
        results.emplace_back(pool.enqueue([i] {
            std::cout << "hello " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "world " << i << std::endl;

            return i * i;
        }));
    }

    for (auto &&result : results)
        std::cout << result.get() << " ";
    std::cout << std::endl;

    for (int i = 1; i < 9; i++) {
        results.emplace_back(pool.enqueue([i] {
            std::cout << "hello " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "world " << i << std::endl;

            return i * i;
        }));
    }

    for (auto &&result : results)
        std::cout << result.get() << " ";
    std::cout << std::endl;

    for (int i = 2; i < 10; i++) {
        results.emplace_back(pool.enqueue([i] {
            std::cout << "hello " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "world " << i << std::endl;

            return i * i;
        }));
    }

    for (auto &&result : results)
        std::cout << result.get() << " ";
    std::cout << std::endl;
}
