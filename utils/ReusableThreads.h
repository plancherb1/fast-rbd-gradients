/**************************************************************************
 *  Reusable Threads
 *
 *  This library is a mashup of threadpools and standard threads. Here we
 *  construct persistent threads that can each be sent specific jobs.
 *  This library is adapted from and inspired by threadpool and llvm. 
 *
 *  threadpool: https://github.com/PaulRitaldato1/ThreadPool
 *  llvm: https://code.woboq.org/llvm/llvm/lib/Support/ThreadPool.cpp.html
 **************************************************************************/

#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <type_traits>
#include <typeinfo>
#include <array>

template <std::size_t NUM_THREADS = 1>
class ReusableThreads{
public:
	// Constructor
	ReusableThreads(){
		shutdown = false;
		ActiveThreads = 0;
		createThreads();
	}

	// Destructor
	~ReusableThreads(){
		shutdown = true;
		for (std::size_t tid = 0; tid < NUM_THREADS; tid++){taskNotifier[tid].notify_all(); threads[tid].join();}
	}

	// Add any arg # function to queue for TID (assume void function)
	template <typename Func, typename... Args>
	void addTask(int tid, Func&& f, Args&&... args){
		auto task = std::make_shared<std::packaged_task<void()>>(std::bind(f, std::forward<Args>(args)...));
		{
			std::unique_lock<std::mutex> lock(taskMutex[tid]); // lock mutex
			taskQueue[tid].emplace([=] {(*task)();}); //place the task into the queue
		}
		taskNotifier[tid].notify_all(); // notify that a job is ready
		return;
	}

	// Wait for all threads to complete and the queue to be empty
	void sync() {
  		std::unique_lock<std::mutex> lock(CompletionLock);
  		// The order of the checks for ActiveThreads and taskQueue[tid].empty() matters because
  		// any active threads might be modifying the jobQueue queue, and this would be a race.
  		CompletionCondition.wait(lock, [&] {
  			bool doneFlag = !ActiveThreads;
  			for (std::size_t tid = 0; tid < NUM_THREADS; tid++){doneFlag &= taskQueue[tid].empty();}
  			return doneFlag;
  		});
	}

private:

	using Job = std::function<void()>;
	std::array<std::thread, NUM_THREADS> threads;
	std::array<std::queue<Job>, NUM_THREADS> taskQueue;
	std::array<std::mutex, NUM_THREADS> taskMutex;
	std::array<std::condition_variable, NUM_THREADS> taskNotifier;
	std::atomic<bool> shutdown;
	std::condition_variable CompletionCondition;
	std::mutex CompletionLock;
	int ActiveThreads;//std::atomic<int> ActiveThreads;

	void createThreads(){
		for (std::size_t tid = 0; tid < NUM_THREADS; tid++){
			threads[tid] = std::thread([this,tid](){
				while (true){
					Job task;
					{
						// wait on a job
						std::unique_lock<std::mutex> lock(taskMutex[tid]);
						taskNotifier[tid].wait(lock, [this,tid] {return !taskQueue[tid].empty() || shutdown;});

						if (shutdown){break;}

						task = std::move(taskQueue[tid].front());
						// Yeah, we have a task, grab it and release the lock on the queue
						// We first need to signal that we are active before popping the queue
						// in order for wait() to properly detect that even if the queue is
						// empty, there is still a task in flight.
						{
							std::unique_lock<std::mutex> lock(CompletionLock);
							++ActiveThreads;
						}
						taskQueue[tid].pop();
					}
					task();
					{
						// Adjust ActiveThreads, in case someone waits on ThreadPool::wait()
						std::unique_lock<std::mutex> lock(CompletionLock);
						--ActiveThreads;
					}
					// Notify task completion, in case someone waits on ThreadPool::wait()
					CompletionCondition.notify_all();
				}
			});
		}
	}
	ReusableThreads (const ReusableThreads&) = delete;
	ReusableThreads& operator= (const ReusableThreads&) = delete;
}; 