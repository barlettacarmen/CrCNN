#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "seal/util/mempool.h"
#include "seal/util/uintarith.h"

using namespace std;

namespace seal
{
    namespace util
    {
        MemoryPoolHeadMT::MemoryPoolHeadMT(int64_t uint64_count) : 
            locked_(false), uint64_count_(uint64_count), 
            alloc_item_count_(mempool_first_alloc_count), 
            first_item_(nullptr)
        {
            allocation new_alloc;

            // Check if we are allocating too much
            int64_t first_alloc_uint64_count = mempool_first_alloc_count * uint64_count_;
            if (first_alloc_uint64_count > mempool_max_batch_alloc_uint64_count)
            {
                throw runtime_error("invalid allocation size");
            }

            try
            {
                new_alloc.data_ptr = new uint64_t[first_alloc_uint64_count];
            }
            catch (const bad_alloc &e)
            {
                // Allocation failed
                throw e;
            }

            new_alloc.size = mempool_first_alloc_count;
            new_alloc.free = mempool_first_alloc_count;
            new_alloc.head_ptr = new_alloc.data_ptr;
            allocs_.clear();
            allocs_.push_back(new_alloc);
        }

        MemoryPoolHeadMT::~MemoryPoolHeadMT()
        {
            bool expected = false;
            while (!locked_.compare_exchange_strong(expected, true, memory_order_acquire))
            {
                expected = false;
            }
            for (size_t i = 0; i < allocs_.size(); i++)
            {
                delete[] allocs_[i].data_ptr;
            }
            allocs_.clear();
            first_item_ = nullptr;
        }

        MemoryPoolItem *MemoryPoolHeadMT::get()
        {
            bool expected = false;
            while (!locked_.compare_exchange_strong(expected, true, memory_order_acquire))
            {
                expected = false;
            }
            MemoryPoolItem *old_first = first_item_;

            // Is pool empty?
            if (old_first == nullptr)
            {
                allocation &last_alloc = allocs_.back();
                MemoryPoolItem *new_item = nullptr;
                if (last_alloc.free > 0)
                {
                    // Pool is empty; there is memory
                    new_item = new MemoryPoolItem(last_alloc.head_ptr);
                    last_alloc.free--;
                    last_alloc.head_ptr += uint64_count_;
                }
                else
                {
                    // Pool is empty; there is no memory
                    allocation new_alloc;

                    // Increase allocation size unless we are already at max
                    int64_t new_size = static_cast<int64_t>(
                        ceil(mempool_alloc_size_multiplier * static_cast<double>(last_alloc.size)));
                    int64_t new_uint64_count = new_size * uint64_count_;
                    if (new_uint64_count > mempool_max_batch_alloc_uint64_count)
                    {
                        new_size = last_alloc.size;
                        new_uint64_count = new_size * uint64_count_;
                    }

                    try
                    {
                        new_alloc.data_ptr = new uint64_t[new_uint64_count];
                    }
                    catch (const bad_alloc &e)
                    {
                        // Allocation failed
                        throw e;
                    }

                    new_alloc.size = new_size;
                    new_alloc.free = new_size - 1;
                    new_alloc.head_ptr = new_alloc.data_ptr + uint64_count_;
                    allocs_.push_back(new_alloc);
                    alloc_item_count_ += new_size;
                    new_item = new MemoryPoolItem(new_alloc.data_ptr);
                }

                locked_.store(false, memory_order_release);
                return new_item;
            }

            // Pool is not empty
            first_item_ = old_first->next();
            old_first->next() = nullptr;
            locked_.store(false, memory_order_release);
            return old_first;
        }

        MemoryPoolHeadST::MemoryPoolHeadST(int64_t uint64_count) :
            uint64_count_(uint64_count), 
            alloc_item_count_(mempool_first_alloc_count), 
            first_item_(nullptr)
        {
            allocation new_alloc;

            // Check if we are allocating too much
            int64_t first_alloc_uint64_count = mempool_first_alloc_count * uint64_count_;
            if (first_alloc_uint64_count > mempool_max_batch_alloc_uint64_count)
            {
                throw runtime_error("invalid allocation size");
            }

            try
            {
                new_alloc.data_ptr = new uint64_t[first_alloc_uint64_count];
            }
            catch (const bad_alloc &e)
            {
                // Allocation failed
                throw e;
            }

            new_alloc.size = mempool_first_alloc_count;
            new_alloc.free = mempool_first_alloc_count;
            new_alloc.head_ptr = new_alloc.data_ptr;
            allocs_.clear();
            allocs_.push_back(new_alloc);
        }

        MemoryPoolHeadST::~MemoryPoolHeadST()
        {
            for (size_t i = 0; i < allocs_.size(); i++)
            {
                delete[] allocs_[i].data_ptr;
            }
            allocs_.clear();
            first_item_ = nullptr;
        }

        MemoryPoolItem *MemoryPoolHeadST::get()
        {
            MemoryPoolItem *old_first = first_item_;

            // Is pool empty?
            if (old_first == nullptr)
            {
                allocation &last_alloc = allocs_.back();
                MemoryPoolItem *new_item = nullptr;
                if (last_alloc.free > 0)
                {
                    // Pool is empty; there is memory
                    new_item = new MemoryPoolItem(last_alloc.head_ptr);
                    last_alloc.free--;
                    last_alloc.head_ptr += uint64_count_;
                }
                else
                {
                    // Pool is empty; there is no memory
                    allocation new_alloc;

                    // Increase allocation size unless we are already at max
                    int64_t new_size = static_cast<int64_t>(
                        ceil(mempool_alloc_size_multiplier * static_cast<double>(last_alloc.size)));
                    if (new_size * uint64_count_ > mempool_max_batch_alloc_uint64_count)
                    {
                        new_size = last_alloc.size;
                    }

                    try
                    {
                        new_alloc.data_ptr = new uint64_t[new_size * uint64_count_];
                    }
                    catch (const bad_alloc &e)
                    {
                        // Allocation failed
                        throw e;
                    }

                    new_alloc.size = new_size;
                    new_alloc.free = new_size - 1;
                    new_alloc.head_ptr = new_alloc.data_ptr + uint64_count_;
                    allocs_.push_back(new_alloc);
                    alloc_item_count_ += new_size;
                    new_item = new MemoryPoolItem(new_alloc.data_ptr);
                }

                return new_item;
            }

            // Pool is not empty
            first_item_ = old_first->next();
            old_first->next() = nullptr;
            return old_first;
        }

        MemoryPoolMT::~MemoryPoolMT()
        {
            WriterLock lock(pools_locker_.acquire_write());
            for (size_t i = 0; i < pools_.size(); i++)
            {
                delete pools_[i];
            }
            pools_.clear();
        }

        Pointer MemoryPoolMT::get_for_uint64_count(int64_t uint64_count)
        {
            if (uint64_count < 0 || uint64_count > mempool_max_single_alloc_uint64_count)
            {
                throw invalid_argument("invalid allocation size");
            }
            else if (uint64_count == 0)
            {
                return Pointer();
            }

            // Attempt to find size.
            ReaderLock reader_lock(pools_locker_.acquire_read());
            size_t start = 0;
            size_t end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                int64_t mid_uint64_count = mid_head->uint64_count();
                if (uint64_count < mid_uint64_count)
                {
                    start = mid + 1;
                }
                else if (uint64_count > mid_uint64_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer(mid_head);
                }
            }
            reader_lock.unlock();

            // Size was not found, so obtain an exclusive lock and search again.
            WriterLock writer_lock(pools_locker_.acquire_write());
            start = 0;
            end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                int64_t mid_uint64_count = mid_head->uint64_count();
                if (uint64_count < mid_uint64_count)
                {
                    start = mid + 1;
                }
                else if (uint64_count > mid_uint64_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer(mid_head);
                }
            }

            // Size was still not found, but we own an exclusive lock so just add it,
            // but first check if we are at maximum pool head count already.
            if (pools_.size() >= static_cast<size_t>(mempool_max_pool_head_count))
            {
                throw runtime_error("maximum pool head count reached");
            }

            MemoryPoolHead *new_head = new MemoryPoolHeadMT(uint64_count);
            if (!pools_.empty())
            {
                pools_.insert(pools_.begin() + start, new_head);
            }
            else
            {
                pools_.emplace_back(new_head);
            }

            return Pointer(new_head);
        }

        int64_t MemoryPoolMT::alloc_uint64_count() const
        {
            ReaderLock lock(pools_locker_.acquire_read());
            int64_t uint64_count = 0;

            for (size_t i = 0; i < pools_.size(); i++)
            {
                MemoryPoolHead *head = pools_[i];
                uint64_count += head->alloc_item_count() * head->uint64_count();
            }
            return uint64_count;
        }

        MemoryPoolST::~MemoryPoolST()
        {
            for (size_t i = 0; i < pools_.size(); i++)
            {
                delete pools_[i];
            }
            pools_.clear();
        }

        Pointer MemoryPoolST::get_for_uint64_count(int64_t uint64_count)
        {
            if (uint64_count < 0 || uint64_count > mempool_max_single_alloc_uint64_count)
            {
                throw invalid_argument("invalid allocation size");
            }
            else if (uint64_count == 0)
            {
                return Pointer();
            }

            // Attempt to find size.
            size_t start = 0;
            size_t end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                int64_t mid_uint64_count = mid_head->uint64_count();
                if (uint64_count < mid_uint64_count)
                {
                    start = mid + 1;
                }
                else if (uint64_count > mid_uint64_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer(mid_head);
                }
            }

            // Size was not found so just add it, but first check if we are at 
            // maximum pool head count already.
            if (pools_.size() >= static_cast<size_t>(mempool_max_pool_head_count))
            {
                throw runtime_error("maximum pool head count reached");
            }

            MemoryPoolHead *new_head = new MemoryPoolHeadST(uint64_count);
            if (!pools_.empty())
            {
                pools_.insert(pools_.begin() + start, new_head);
            }
            else
            {
                pools_.emplace_back(new_head);
            }

            return Pointer(new_head);
        }

        int64_t MemoryPoolST::alloc_uint64_count() const
        {
            int64_t uint64_count = 0;
            for (size_t i = 0; i < pools_.size(); i++)
            {
                MemoryPoolHead *head = pools_[i];
                uint64_count += head->alloc_item_count() * head->uint64_count();
            }
            return uint64_count;
        }
    }
}
