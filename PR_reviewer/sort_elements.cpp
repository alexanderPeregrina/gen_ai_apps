
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Initialize vector with the given elements
    std::vector<int> numbers = {10, 20, 5, 6, 8, 9, -4, 8, 1};
    
    // Sort the vector in ascending order
    std::sort(numbers.begin(), numbers.end());
    
    // Print the sorted elements
    std::cout << "Sorted elements: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
