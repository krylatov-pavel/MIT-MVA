from utils.bitmask import to_bitmask

class Combinator(object):
    def __init__(self, accuracy=0.005):
        self.accuracy = accuracy

    def split(self, groups, split_ratio):
        """Split groups into k subgroups with given split_ratio
        Args:
            elements: list of tuples, first el is "name", second is "count" e.g [("418", 101), ("419", 3), ...]
            split_ratio: k-length list of float, k is the number of subgroups, k[i] is relative size
            of the i-th subgroup, e.g [0.2, 0.2, 0.2, 0.2, 0.2] for 5-fold cross validation
        Returns:
            k-length list of lists of group names, e.f [["418", "419"], ["500"], ...]
        """
        splits = []

        #TO DO: if groups length is more then, say, 15, reduce groups number
        #by combining several small groups into a bigger one

        remaining_groups = groups.copy()

        for i in range(len(split_ratio) - 1):
            #relative to remaining groups
            ratio = split_ratio[i] / sum(split_ratio[i:])

            subgroup = self._get_subgroup(remaining_groups, ratio)
            splits.append(subgroup)

            remaining_groups = [el for el in remaining_groups if not (el in subgroup)]

        splits.append(remaining_groups)

        return splits

    def _get_subgroup(self, elements, ratio):
        """Picks subgroup of elements taking into account given ratio
        Args:
            elements: list of tuples, first el is "name", second is "count" e.g [("418", 101), ("419", 3), ...]
            ratio: float, relative size of subgroup, e.g 0.2
        Returns:
            list of indexes of elemets in subgroup, e.g [0, 10, 11]
        """
        
        bits = len(elements)
        total_size = sum(el[1] for el in elements)

        best_combination = [0] * bits
        best_combination_error = 1.0

        for i in range(1, 2 ** bits):
            combination = to_bitmask(i, bits)
            subgroup_size = self.__subgroup_size(elements, combination)
            curr_error = abs(subgroup_size / total_size - ratio)
            if curr_error < best_combination_error:
                best_combination_error = curr_error
                best_combination = combination.copy()
                if curr_error <= self.accuracy:
                    #consider this result as "good enough", don't waste calculation time
                    break
        
        return [el for el, include in zip(elements, best_combination) if include == 1]

    def __subgroup_size(self, elements, mask):
        return sum(el[1] for el, include in zip(elements, mask) if include == 1)         