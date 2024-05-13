from functools import wraps
import time

# Global variable to store the log file path
log_file_path = None

# Initialize the dictionary
execution_times = {}

def time_tracker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Accumulate the time for each function
        if func.__name__ in execution_times:
            execution_times[func.__name__] += elapsed_time
        else:
            execution_times[func.__name__] = elapsed_time

        return result
    return wrapper

# Function to display the accumulated times
def display_execution_times():
    total = 0
    for func_name, total_time in execution_times.items():
        print(f"Total execution time for {func_name}: {total_time:.4f} seconds")
        total += total_time
    print(f"Total execution time for all functions: {total:.4f} seconds")
    return total

# Global dictionary to store call counts
call_counts = {}

def call_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Increment the count for each function call
        call_counts[func.__name__] = call_counts.get(func.__name__, 0) + 1
        return func(*args, **kwargs)

    return wrapper

# Function to display the call counts
def display_call_counts():
    for func_name, count in call_counts.items():
        print(f"{func_name} was called {count} times")

# Function to save the tracker values to a log file
def save_tracker_logs():
    if log_file_path:
        with open(log_file_path, 'w') as log_file:
            log_file.write("Call Counts:\n")
            for func_name, count in call_counts.items():
                log_file.write(f"{func_name}: {count}\n")

            log_file.write("\nExecution Times:\n")
            total = 0
            for func_name, total_time in execution_times.items():
                log_file.write(f"{func_name}: {total_time:.2f} seconds\n")
                total += total_time
            log_file.write(f"Total execution time for all functions: {total:.2f} seconds\n")
                
    else:
        print("Log file path not specified. Skipping saving tracker logs.")