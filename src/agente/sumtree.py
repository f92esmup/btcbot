"""
SumTree Data Structure for Prioritized Experience Replay

This module implements a SumTree, which is a binary tree where each leaf stores 
a priority value and each parent node contains the sum of its children. This 
structure enables efficient O(log N) sampling based on priorities, making it 
ideal for Prioritized Experience Replay in reinforcement learning.

Author: GitHub Copilot
"""

import numpy as np
from typing import Tuple


class SumTree:
    """
    A SumTree data structure for efficient priority-based sampling.
    
    The SumTree is implemented as a complete binary tree stored in an array.
    Leaf nodes store priority values, and internal nodes store the sum of 
    their children's values. This allows for O(log N) sampling and updates.
    
    Tree structure:
    - Array indices 0 to capacity-1: internal nodes
    - Array indices capacity-1 to 2*capacity-2: leaf nodes (priorities)
    - Root is at index 0
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the SumTree with a given capacity.
        
        Args:
            capacity (int): Maximum number of leaf nodes (data points) the tree can store.
        """
        self.capacity = capacity
        self.write_pointer = 0
        
        # Tree array: internal nodes + leaf nodes
        # Size: 2 * capacity - 1
        # Indices [0, capacity-1): internal nodes
        # Indices [capacity-1, 2*capacity-1): leaf nodes
        self.tree = np.zeros(2 * capacity - 1)
        
        # Data array to store actual data indices associated with each leaf
        self.data = np.zeros(capacity, dtype=int)
    
    def _propagate(self, tree_index: int, change: float) -> None:
        """
        Propagate priority changes up the tree to maintain correct sums.
        
        This method updates all parent nodes from the given tree_index up to 
        the root to reflect a change in priority.
        
        Args:
            tree_index (int): Index in the tree array where the change occurred.
            change (float): The change in priority value to propagate.
        """
        # Get parent index
        parent_index = (tree_index - 1) // 2
        
        # Update parent with the change
        self.tree[parent_index] += change
        
        # Continue propagating up if we haven't reached the root
        if parent_index != 0:
            self._propagate(parent_index, change)
    
    def _retrieve(self, tree_index: int, sample_value: float) -> int:
        """
        Retrieve the leaf index corresponding to a sample value.
        
        This method performs a binary search down the tree to find the leaf
        that corresponds to the given sample value. The search is based on
        the cumulative sum of priorities.
        
        Args:
            tree_index (int): Current node index in the tree (starts at root = 0).
            sample_value (float): Value to search for (0 <= sample_value < total_priority).
            
        Returns:
            int: Index of the leaf node in the tree array.
        """
        # Calculate left and right child indices
        left_child = 2 * tree_index + 1
        right_child = left_child + 1
        
        # If we've reached a leaf node, return its index
        if left_child >= len(self.tree):
            return tree_index
        
        # Decide which subtree to search based on sample_value
        if sample_value <= self.tree[left_child]:
            # Search left subtree
            return self._retrieve(left_child, sample_value)
        else:
            # Search right subtree, adjusting sample_value
            return self._retrieve(right_child, sample_value - self.tree[left_child])
    
    def add(self, priority: float, data_index: int) -> None:
        """
        Add a new priority and its associated data index to the tree.
        
        This method adds a new priority at the current write_pointer position,
        updates the tree structure, and advances the write pointer.
        
        Args:
            priority (float): Priority value for the new data point.
            data_index (int): Index of the actual data in the main buffer.
        """
        # Calculate tree index for the current leaf position
        tree_index = self.write_pointer + self.capacity - 1
        
        # Store the data index
        self.data[self.write_pointer] = data_index
        
        # Update the tree with the new priority
        self.update(tree_index, priority)
        
        # Advance write pointer (circular buffer)
        self.write_pointer = (self.write_pointer + 1) % self.capacity
    
    def update(self, tree_index: int, new_priority: float) -> None:
        """
        Update the priority of an existing leaf and propagate the change.
        
        Args:
            tree_index (int): Index of the leaf in the tree array to update.
            new_priority (float): New priority value for the leaf.
        """
        # Calculate the change in priority
        change = new_priority - self.tree[tree_index]
        
        # Update the leaf with the new priority
        self.tree[tree_index] = new_priority
        
        # Propagate the change up the tree
        self._propagate(tree_index, change)
    
    def get(self, sample_value: float) -> Tuple[int, float, int]:
        """
        Sample from the tree based on a given sample value.
        
        This is the main sampling method that finds the leaf corresponding
        to the sample value and returns information about it.
        
        Args:
            sample_value (float): Value to sample (0 <= sample_value < total_priority).
            
        Returns:
            Tuple[int, float, int]: A tuple containing:
                - tree_index: Index of the leaf in the tree array
                - priority: Priority value of the sampled leaf
                - data_index: Index of the data in the main buffer
                
        Raises:
            ValueError: If sample_value is out of valid range.
        """
        if sample_value < 0 or sample_value >= self.total_priority:
            raise ValueError(f"Sample value {sample_value} out of range [0, {self.total_priority})")
        
        # Find the leaf index
        leaf_index = self._retrieve(0, sample_value)
        
        # Get the priority of the leaf
        priority = self.tree[leaf_index]
        
        # Calculate the data index (convert tree leaf index to data array index)
        data_array_index = leaf_index - self.capacity + 1
        data_index = self.data[data_array_index]
        
        return leaf_index, priority, data_index
    
    @property
    def total_priority(self) -> float:
        """
        Get the total sum of all priorities in the tree.
        
        Returns:
            float: Sum of all priority values (value at root of tree).
        """
        return self.tree[0]
    
    def __len__(self) -> int:
        """
        Get the current number of elements in the tree.
        
        Returns:
            int: Number of elements currently stored.
        """
        # Count non-zero priorities in the leaf nodes
        leaf_start = self.capacity - 1
        leaf_end = 2 * self.capacity - 1
        return int(np.count_nonzero(self.tree[leaf_start:leaf_end]))
    
    def __repr__(self) -> str:
        """
        String representation of the SumTree.
        
        Returns:
            str: String representation showing capacity and current size.
        """
        return f"SumTree(capacity={self.capacity}, size={len(self)}, total_priority={self.total_priority:.2f})"
