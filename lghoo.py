#https://github.com/davidissamattos/LG-HOO
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import os.path
import sys
from savitzky_golay import *

class LGHOO(object):
    def __init__(self, arm_range, height_limit=10, v1=1.0, rho=0, minimum_grow=10):
        """
        Initializing a list of arms
        :param arm_range: a vector with the two borders for the whole range. It can be either [5,1] or [1,5] for example
        :param height_limit: limits the height of the tree. In practicality it limits the size of the vector to 2^height + 2 of the borders.
        The default value is 10

        """
        if minimum_grow > 0:
            self.minimum_grow = minimum_grow # v1 should be greater than 0
        else:
            self.minimum_grow = 10
        self.arm_range = arm_range
        self.range_size = np.absolute(arm_range[0] - arm_range[1])
        self.midrange_arm = self.range_size/2
        self.arm_range_min = np.min(self.arm_range)
        self.arm_range_max = np.max(self.arm_range)
        self.height_limit = height_limit
        self.initial_bound = np.inf
        self.initial_children_arm = np.nan
        if v1 > 0:
            self.v1 = v1 # v1 should be greater than 0
        else:
            self.v1 = 1.0
            print("v1 should be greater than 0")
        # rho should be < 1
        if rho < 1:
            self.rho = rho

        else:
            self.rho = 0.5
            print("rho should be less than 1")


        # the vector that we will keep track representing all our optimization space
        #   [0            1       2          3            4         5         6                   7               8                  9         ]
        #  [[value_arm, bound, ncounts, mean_value, sum_rewards, height, parent_arm_value, left_children, right_children, decision_arm_criteria]]


        # in height 1 the parent is itself
        # border nodes
        #self.arm_list = np.array([[self.arm_range_min, self.initial_bound, 0, 0, 0, 1, self.arm_range_min,
        #                           self.initial_children_arm, self.initial_children_arm]])
        #self._add_arm([[self.arm_range_max, self.initial_bound, 0, 0, 0, 1, self.arm_range_max,
        #               self.initial_children_arm, self.initial_children_arm]])
        # root node

        self.arm_list = np.array([[self.midrange_arm, self.initial_bound, 0, 0, 0, 1, self.range_size / 2,
                       self.initial_children_arm, self.initial_children_arm, 0]])
        self.arm_list_names = ["arm_value", "bound_value", "ncounts", "mean", "sum_rewards","height", "parent_arm", "left_children_arm", "right_children_arm" ]

# Some tree functions
## Adding values to the list
    def _add_child_for_arm(self, arm):
        """
        Add a child for in the tree
        :param arm:
        :return:
        """

        index = self._get_index_for_arm(arm)
        arm_height = self._get_height_for_index(index)

        #making sure we dont go over the height limit
        if arm_height >= self.height_limit:
            return
        # Making sure we dont add repetitive children
        #if has a child we skip the function else we add both children

        if self._has_child_for_index(index):
            #print "already have a child"
            return
        else:
            n = self._get_number_of_arms()
            child_height = arm_height + 1
            diff = self.range_size / np.power(2, child_height)

            # adding the child arms
            left_child_arm = (arm - diff)
            right_child_arm = (arm + diff)

            # child to the left
            self.arm_list[index][7] = left_child_arm  # setting the right children value in the parent
            # child to the right
            self.arm_list[index][8] = right_child_arm  # setting the right children value in the parent
            # adding at the end so we dont need to get another index
            self._add_arm([[left_child_arm, self.initial_bound, 0, 0, 0, child_height, arm, self.initial_children_arm,
                            self.initial_children_arm, 0]])
            self._add_arm([[right_child_arm, self.initial_bound, 0, 0, 0, child_height, arm, self.initial_children_arm,
                            self.initial_children_arm, 0]])
            return

    def _add_child_for_index(self, index):
        """
        Use the main add_child)for arm function
        :param index:
        :return:
        """
        arm = self._get_arm_value_for_index(index)
        return self._add_child_for_arm(arm)

    def _add_arm(self, arm_vector):
        self.arm_list = np.append(self.arm_list, arm_vector, axis=0)
        #sort by arm
        #self._sort_list_arm_by_arm()

    def _has_child_for_index(self, index):
        """
        Checking to see if a node has any children
        :param index: index of the arm
        :return: True if one of the children is not nan and false if both are nan
        """
        left_child = self.arm_list[index][7]
        right_child = self.arm_list[index][8]
        # we want to be sure that both children are nan
        if np.isnan(left_child) and np.isnan(right_child):
            return False
        else:
            return True

    def _has_child_for_arm(self, arm):
        """
        Checking to see if a node has any children
        :param arm: the value of the arm
        :return: True if one of the children is not nan and false if both are nan
        """
        index = self._get_index_for_arm(arm)
        return self._has_child_for_index(index)

    def _get_left_child_arm_for_index(self, index):
        return self.arm_list[index][7]

    def _get_right_child_arm_for_index(self, index):
        return self.arm_list[index][8]

    def _get_left_child_index_for_index(self, index):
        arm = self._get_left_child_arm_for_index(index)
        if np.isnan(arm):
            return np.nan
        else:
            return self._get_index_for_arm(arm)

    def _get_right_child_index_for_index(self, index):
        arm = self._get_right_child_arm_for_index(index)
        if np.isnan(arm):
            return np.nan
        else:
            return self._get_index_for_arm(arm)

    def _get_root_index(self):
        return self._get_index_for_arm(self.midrange_arm)

    def _sort_list_arm_by_height(self):
        """
        Returns and array sorted by the height. Highest height goes first
        :return:
        """
        self.arm_list = np.flipud(self.arm_list[self.arm_list[:, 5].argsort()])

    def _sort_list_arm_by_arm(self):
        self.arm_list = self.arm_list[self.arm_list[:, 0].argsort()]

## Get the value for the height of the node
    def _get_height_for_index(self, index):
        return self.arm_list[index][5]

    def _get_height_for_arm(self, arm):
        index = self._get_index_for_arm(arm)
        return self._get_height_for_index(index)

    def _get_max_height(self):
        return np.max(self.arm_list[:,5])

    def _get_tree_height(self):
        return self.arm_list[:, 5].max()

## Get value for paths and searching the tree
    def _get_parent_arm_for_child_index(self, index):
        """
        Return the value of the arm of the parent given a child index
        :param index:
        :return:
        """
        return self.arm_list[index][6]

    def _get_parent_index_for_child_index(self, index):
        """
        Return the index of the parent given a child index
        :param index:
        :return:
        """
        return self._get_index_for_arm(self._get_parent_arm_for_child_index(index))

    def _get_parent_arm_for_child_arm(self, arm):
        """
        return the value of the arm given a child arm
        :param arm:
        :return:
        """
        index = self._get_index_for_arm(arm)
        return self._get_parent_arm_for_child_index(index)

    def _get_parent_index_for_child_arm(self, arm):
        """
        Return the index of the parent given the child arm
        :param arm:
        :return:
        """
        index = self._get_index_for_arm(arm)
        return self._get_parent_index_for_child_index(index)

    def _get_path_for_index(self, index):
        """
        receives an index and returns a numpy array of index for the path
        Searches through the list to find the whole path
        :return:
        """
        list_index = np.array([index])
        height = self._get_height_for_index(index).astype(np.int64)
        for i in range(1, height):
            node_index = self._get_parent_index_for_child_index(list_index[-1])
            list_index = np.append(list_index, node_index)
        return list_index

    def _get_path_for_arm(self, arm):
        """
        receives an arm and returns a numpy array of index for the path
        :param arm:
        :return:
        """
        index = self._get_index_for_arm(arm)
        return self._get_path_for_index(index)

## Get/Update the values for the bounds

    def _get_bound_for_index(self, index):
        return self.arm_list[index][1]

    def _set_bound_for_index(self, index, bound):
        self.arm_list[index][1] = bound

    def _get_bound_for_arm(self, arm):
        index = self._get_index_for_arm(arm)
        return self._get_bound_for_index(index)

    def _get_number_of_max_bound(self):
        return self._get_all_index_max_bound().size

    def _get_max_bound_arm(self):
        index = self._get_max_bound_index()
        return self._get_arm_value_for_index(index)

    def _get_max_bound_index(self):
        return self._get_all_index_max_bound()[0][0]

    def _get_most_promising_leaf_index(self):
        path = self._get_index_path_for_possible_arms()
        return path[-1]

    def _set_bound_for_arm(self, arm, bound):
        index = self._get_index_for_arm(arm)
        self._set_bound_for_index(index, bound)

    def _get_all_index_max_bound(self):
        """
        Return an array with the index of all max bounds if there is more than one
        :return:
        """
        return np.argwhere(self.arm_list[:, 1] == np.amax(self.arm_list[:, 1]))

    def _debug_arms_and_bounds(self):
        index = self.get_best_arm_index()
        bounds = self._get_bound_for_index(index)
        arms = self._get_arm_value_for_index(index)
        count = self._get_count_for_index(index)
        return arms, bounds, count

    def _old_get_most_promising_child(self):
        """
        Returns uniformly random the index of the max bound and create a child for this node
        :return: the index of the most promising child
        """
        bounds = self._get_all_index_max_bound()
        choose = np.random.choice(bounds.shape[0])

        index_most_promising_child = bounds[choose][0]

        return index_most_promising_child

    def _get_index_path_for_possible_arms(self):
        """We return an array with indexes with arms that we can choose"""
        root_index = self._get_index_for_arm(self.midrange_arm)
        path=[]
        arms=[]
        arms.append(self._get_arm_value_for_index(root_index))
        path.append(root_index)
        iter_index = root_index
        #while we have children
        children = True
        while children:
            parent_index = iter_index
            iter_index = self._get_highest_bound_child_for_index(parent_index)
            if np.isnan(iter_index):
                children = False
            else:
                path.append(iter_index)
                arms.append(self._get_arm_value_for_index(iter_index))

        #print self.print_full_list()
        #print "arms path", arms
        return np.array(path)

    def _get_highest_bound_child_for_index(self, index):

        bound_left = []
        bound_right = []

        left_index = self._get_left_child_index_for_index(index)

        right_index = self._get_right_child_index_for_index(index)

        most_promising_child = []
        #if haas left child
        if not np.isnan(left_index):
            bound_left = self._get_bound_for_index(left_index)
        else:
            return np.nan
        # if haas right child
        if not np.isnan(right_index):
            bound_right = self._get_bound_for_index(right_index)
        else:
            return np.nan

        if bound_right > bound_left:
            most_promising_child = right_index
        if bound_left > bound_right:
            most_promising_child = left_index
        if bound_right == bound_left:
            choose = np.random.choice(2)
            if choose == 1:
                most_promising_child = right_index
            if choose == 0:
                most_promising_child = left_index

        return most_promising_child

    def _extend_tree_for_index(self, index):
        """
        Extend the tree in the index

        """
        if self._get_count_for_index(index) >= self.minimum_grow:
            self._add_child_for_index(index)
        else:
            return

    def _extend_tree_for_arm(self, arm):
        index = self._get_index_for_arm(arm)
        self._extend_tree_for_index(index)

    def _find_nearest_node_by_arm(self, arm):
        """
        This function works only in case the player plays an arm that was not previously allocated or thought in the list of possible arms
        In this case we take the tree closest arm to the played arm. If the difference between the two would be greater than of the childs we discard the vector
        else we store it in the found position
        :param arm:
        :return: the index
        """
        min_index = (np.abs(self.arm_list[:,0] - arm)).argmin()
        diff = np.abs(self.arm_list[min_index,0]-arm)

        arm_height = self._get_height_for_index(min_index)
        child_height = arm_height + 1
        child_diff = self.range_size / np.power(2, child_height)

        if diff > child_diff:
            return -1
        else:
            return min_index


    def _calculate_ucb_for_index(self, index):
        """
        Calculate the ucb estimate for an element
        :param index:
        :return:
        """
        counts = self._get_count_for_index(index)
        if counts > 0:
            uncertainty_bound = math.sqrt((2 * math.log(self._get_total_counts())) / float(self._get_count_for_index(index)))
            mean = self._get_mean_for_index(index)
            h = self._get_height_for_index(index)
            max_variation_payoff = self.v1 * np.power(self.rho, h)
            ucb_value = mean + uncertainty_bound + max_variation_payoff
        else:
            ucb_value = np.inf
        return ucb_value

    def _calculate_bound_for_index(self, index):
        """
        Calculate the final Bound or B value for the index and save in the bound part of the vector arm_list
        in col index 1
        :param index:
        :return:
        """
        ucb = self._calculate_ucb_for_index(index)
        left_children_bound = np.inf
        right_children_bound = np.inf

        #if it has a child it updates the values
        if self._has_child_for_index(index):
            left_children_index = self._get_left_child_index_for_index(index)
            left_children_bound = self._get_bound_for_index(left_children_index)

            right_children_index = self._get_right_child_index_for_index(index)
            right_children_bound = self._get_bound_for_index(right_children_index)
        max_bound_children = np.max(np.array([left_children_bound, right_children_bound]))
        bound = np.min(np.array([ucb, max_bound_children]))
        self._set_bound_for_index(index, bound)

    def _calculate_arm_decision_criteria_for_index(self, index):
        """Decision criteria"""
        #This one is ok
        # bound = self._get_bound_for_index(index)
        # if np.isnan(bound) or np.isinf(bound):
        #     self.arm_list[index, 9] = 0
        # else:
        #     self.arm_list[index,9] = self._get_bound_for_index(index)/self._calculate_ucb_for_index(index)
        #
        #this is better
        bound = self._get_bound_for_index(index)
        if np.isnan(bound) or np.isinf(bound):
            self.arm_list[index, 9] = 0
        else:
            self.arm_list[index,9] = self._get_mean_for_index(index) / self._calculate_ucb_for_index(index)

    def _update_all_bounds(self):
        """
        In this function we update all bounds
        First we sort by height
        Then we update it sequentially
        we dont calculate for the borders
        :return:
        """
        narms = self._get_number_of_arms()
        self._sort_list_arm_by_height()
        for i in range(narms):
            self._calculate_bound_for_index(i)
            self._calculate_arm_decision_criteria_for_index(i)

## Get/Update the values for the counts


## Get counts

    def _get_count_for_index(self, index):
        return self.arm_list[index][2]

    def _get_count_for_arm(self, arm):
        index = self._get_index_for_arm(arm)
        return self._get_count_for_index(index)

    def _update_n_count_for_index(self, index, n):
        self.arm_list[index][2] = self.arm_list[index][2] + n

    def _update_n_count_for_arm(self, arm, n):
        index = self._get_index_for_arm(arm)
        self._update_n_count_for_index(index, n)

    def _update_count_for_arm(self, arm):
        self._update_n_count_for_arm(arm, 1)

    def _update_count_for_index(self, index):
        self._update_n_count_for_index(index, 1)

 ## Get/Update the Sum of reward values
    def _get_reward_for_index(self, index):
        return self.arm_list[index][4]

    def _get_reward_for_arm(self, arm):
        index = self._get_index_for_arm(arm)
        return self._get_reward_for_index(index)

    def _update_reward_for_index(self, index, sum_rewards):
        self.arm_list[index][4] = self.arm_list[index][4] + sum_rewards

    def _update_reward_for_arm(self, arm, sum_rewards):
        index = self._get_index_for_arm(arm)
        self._update_reward_for_index(index, sum_rewards)

## Get/Update value for the mean values

    def _get_mean_for_index(self, index):
        return self.arm_list[index][3]

    def _get_mean_for_arm(self, arm):
        index = self._get_index_for_arm(arm)
        return self._get_mean_for_index(index)

    def _update_n_mean_for_index(self, index, n, sum_new_rewards):
        """
        Update the mean and the counts for the played arm
        :param index:
        :param n:
        :param sum_new_rewards:
        :return:
        """
        self._update_reward_for_index(index, sum_new_rewards)

        n_old = self._get_count_for_index(index)
        #update the count for the played arm
        self._update_n_count_for_index(index, n)
        n_new = self._get_count_for_index(index)

        old_mean = self._get_mean_for_index(index)

        new_mean = (old_mean * n_old + sum_new_rewards) / float(n_new)
        self.arm_list[index][3] = new_mean

    def _update_mean_for_index(self, index, reward):
        self._update_n_mean_for_index(index=index, n=1, sum_new_rewards=reward)

    def _update_n_mean_for_arm(self, arm, n, sum_new_rewards):
        index = self._get_index_for_arm(arm)
        self._update_n_mean_for_index(index=index, n=n, sum_new_rewards=sum_new_rewards)

    def _update_mean_for_arm(self, arm, reward):
        self._update_n_mean_for_arm(arm=arm, n=1, sum_new_rewards=reward)

    def _refresh_all_means(self):
        """
        Dividing the rewards by the number of counts element wise
        :return:
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            self.arm_list[:, 3] = np.divide(self.arm_list[:, 4], self.arm_list[:, 2])
            self.arm_list[np.isnan(self.arm_list)] = 0

        # Other functions

## value of the arms functions

    def _get_number_of_arms(self):
        """
        Number of active arms
        :return: the number of rows -> index 0
        """
        return self.arm_list.shape[0]

    def _get_arm_value_for_index(self, index):
        return self.arm_list[index][0]

    def _get_index_for_arm(self, arm):
        """
        Return the index for a specific arm
        :param arm: value of the arm
        :return: return an array
        """
        return np.where(self.arm_list[:, 0] == arm)[0][0]

    def _get_all_arms_from_arm_list(self):
        return self.arm_list[:, 0]

    def _get_full_arm_list(self):
        return self.arm_list

    def _get_total_counts(self):
        return np.sum(self.arm_list[:, 2])

## Algorithm part
    def select_arm(self):
        """
        Selects an arm to play and return the value of the arm
        :return:
        """
        # 1 - Select the path of the most promising children
        path = self._get_index_path_for_possible_arms()

        # Select randomly an arm from the path of the most promising child
        chosen_arm_index = path[np.random.choice(path.size)]
        #Return the value of the arm
        print(self._get_arm_value_for_index(chosen_arm_index))
        return self._get_arm_value_for_index(chosen_arm_index)

    def update(self, chosen_arm, reward):
        """
        For the simulation framework. The chosen arm is the value of the arm in the vecotr
        :param chosen_arm: value of the arm
        :param reward: received reward
        :return:
        """

        #Extend tree with the new bounds (inf) for the child
        index = self._find_nearest_node_by_arm(chosen_arm)
        #if the chosen arm is not valid we ignore it
        if index == -1:
            #print "no arm selected"
            return
        else:

            arm_to_be_updated = self._get_arm_value_for_index(index)
            #Update for all the path
            path = self._get_path_for_arm(arm_to_be_updated)
            for i in path:
                self._update_mean_for_index(index=i, reward=reward)

            print("arm played: ", chosen_arm, " arm: ", arm_to_be_updated, " selected")
            print('reward', reward)
            self._extend_tree_for_arm(arm_to_be_updated)
            #update statistics for the node
            #self._update_mean_for_arm(arm=arm_to_be_updated, reward=reward)
            #Update the computation of all bounds in the tree
            self._update_all_bounds()
            return


    def get_original_best_arm_index(self):
        """
        The original way in the paper
        :return:
        """
        return np.argwhere(self.arm_list[:, 5] == np.amax(self.arm_list[:, 5]))[0][0]

    def get_original_best_arm(self):
        """
        The original way in the paper
        :return:
        """
        index = self.get_original_best_arm_index()
        return self._get_arm_value_for_index(index)


    def get_best_arm_index(self):
        """
        Based on the criteria vector
        :return:
        """
        return np.argwhere(self.arm_list[:, 9] == np.amax(self.arm_list[:, 9]))[0][0]

    def get_best_arm_value(self):
        """
        Return the value of the best arm based on the criteria
        :return:
        """
        index = self.get_best_arm_index()
        return self._get_arm_value_for_index(index)

## Visualization functions

    def print_full_list(self):
        print(self.arm_list_names)
        print(self._get_full_arm_list())

    def data_to_plot(self):
        G =self.get_graph()
        pos = nx.get_node_attributes(G, 'pos')
        return G, G.nodes, G.edges, pos

    def get_graph(self):
        G = nx.Graph()
        n = self._get_number_of_arms()
        for i in range(n):
            G.add_node(self.arm_list[i, 0], pos=(self.arm_list[i, 0], self.arm_list[i, 5]))
            G.add_edge(float(self.arm_list[i, 0]), self.arm_list[i, 6])
        return G

    def plot_graph(self):
        G = self.get_graph()
        pos = nx.get_node_attributes(G, 'pos')
        plt.title('draw_networkx')
        nx.draw(G, pos=pos, with_labels=True)
        plt.show()


    def plot_graph_with_function(self,x_axis, y_axis,rescale_y=1,save=False,filename ="output.png"):
        fig, ax = plt.subplots()
        G = self.get_graph()
        pos = nx.get_node_attributes(G, 'pos')

        max_h = int(self._get_max_height())
        function_y0 = max_h+ 1
        max_y = function_y0 + rescale_y + 0.5
        best_arm_x = self.get_best_arm_value()
        text_x_offset = 0.01*(self.arm_range_max-self.arm_range_min)
        text_y_offset = -0.5


        #plotting the estimated underlying function
        try:
            self._sort_list_arm_by_arm()
            est_x = self.arm_list[:,0]
            number_arms = self.arm_list.shape[0]
            window = convert2odd(number_arms/2)
            est_y = savitzky_golay((self.arm_list[:,9]),window_size=window,order=max_h)*rescale_y+ function_y0
            plt.plot(est_x,est_y, label="Estimated function")
        except:
            pass

        #ploting the function after the max height
        func_y = rescale_y*y_axis+function_y0
        func_x = x_axis
        ax.plot(func_x, func_y, label="True function")

        #Drawing the graph
        nx.draw(G, pos=pos, node_size=20,ax=ax)

        # Line of the best arm
        plt.axvline(x=best_arm_x, linestyle='--')
        ax.text(best_arm_x + text_x_offset, function_y0 + text_y_offset, "Best arm:\n" + str(format(best_arm_x, '.4f')))


        # Line for the heights
        for i in range(1,max_h+1):
            plt.axhline(y=i, linestyle='--', linewidth=0.5)
            ax.text(0,i,str(i), size='smaller')

        #line for the zero in the underlying function
        plt.axhline(y=function_y0, linestyle='-',linewidth=0.5, color='black')
        plt.axhline(y=function_y0+rescale_y, linestyle='-', linewidth=0.5, color='black')

        #Setting figure style
        #axis on but just with the x axis, the y axis is off
        plt.axis('on')
        plt.xlim(self.arm_range_min,self.arm_range_max)
        ax.get_yaxis().set_visible(False)
        plt.ylim([1,max_y])

        #setting title for the plot and axis
        ax.set_title('Best value optimization using LG-HOO')
        ax.set_xlabel('Values in the $\chi$-space')

        lgd = ax.legend(loc = 9, bbox_to_anchor=(0.5,-0.15),ncol=2)
        fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
#
        if save == True:
            mainfile = sys.argv[0]
            pathname = os.path.join(os.path.dirname(mainfile),"pictures")
            output = os.path.join(pathname,filename)
            plt.savefig(output, format='png', dpi=300)#, bblock_extra_artists=(lgd,), bbox_inches='tight')
        else:
            #plt.tight_layout()
            plt.show()

    def save_list_arms_to_file(self):
        np.savetxt("data.csv", self._get_full_arm_list(), delimiter=",")