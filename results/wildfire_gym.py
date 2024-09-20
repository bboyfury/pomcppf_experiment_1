import sys; sys.path.append("../..")

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireGymSimulation

import numpy as np
import random
import torch
from torch_geometric.data import Data


AGENT_NODE_NUMBER = 1
TASK_NODE_NUMBER = 2
ACTION_NODE_NUMBER = 3


class WildfireGymEnvironment:
    def __init__(self, run, setup, filename_postfix, training_mode):
        # save the params
        self.run = run
        self.setup = setup
        self.filename_postfix = filename_postfix
        self.training_mode = training_mode

        # create the rest of the attributes
        self.reset()


    def reset(self):
        # create the simulator
        self.simulator = WildfireGymSimulation(self.run, self.setup, self.filename_postfix)
        self.domain = self.simulator.domain
        self.domain_settings = self.domain.settings
        self.num_agents = self.simulator.num_agents()
        self.num_fires = self.simulator.num_fires()

        # get the initial state info
        initial_info = self.simulator.initialize()
        self.state = initial_info[0]
        self.state_index = initial_info[1]
        self.last_actions = []
        self.last_observations = []
        self.last_rewards = []

        # log if necessary
        if self.training_mode:
            self.step_num = -1
        else:
            self.simulator.log_initial(self.state, self.state_index)
            self.step_num = 1

        # create the action space
        # actions 0 through n-1 are for the n fires, the last action is NOOP
        self.action_space = []
        for agent_i in range(self.num_agents):
            agent = self.simulator.domain.agents[agent_i]
            frame = agent.frame
            self.action_space.append(list(frame.all_available_actions))

        return self.state


    def step(self, actions):
        # take a single step of the simulation
        ret = self.simulator.step(self.state, self.state_index, actions)

        # parse the step information
        next_state = ret[0]
        next_state_index = ret[1]
        observations_list = ret[2]
        rewards_list = ret[3]

        # log if necessary
        if not self.training_mode:
            self.simulator.log(self.step_num, actions, next_state, next_state_index,
                               observations_list, rewards_list)
            self.step_num += 1

        # update our information
        self.state = next_state
        self.state_index = next_state_index
        self.last_actions = actions
        self.last_observations = observations_list
        self.last_rewards = rewards_list

        # return the information
        return self.state, rewards_list, observations_list


    def getObsFromState(self, state, know_all_suppressants=False):
        observations = []

        for agent_i in range(self.num_agents):
            if know_all_suppressants:
                # the agent knows everyone else's suppressant levels, so they get the entire state
                observation = list(state)
            else:
                # copy the fully observable fire information
                observation = list(state[:self.num_fires])

                # the agent only knows its own suppressant levels
                observation.append(state[self.num_fires + agent_i])

            observations.append(observation)

        return observations


    def create_graph_base(self, state_or_obs, agent_idx=-1, know_all_suppressants=False):
        '''
        Function to convert an observation (or state, if agent_idx == -1) into an interaction graph
        '''

        '''
        Node Features:
        Agent node: [node_type (1), agent location x, agent location y, fire reduction power, suppressant level]
                    Note: suppressant_level of -1 implies unknown (also used if agent_idx == -1)
        Task node: [node_type (2), fire location x, fire location y, intensity, empty]
        Action node: [node_type (3), action type, empty, empty, empty]
        Edge node: [node_type (0), empty, empty, empty, empty]
        
        action type: [fire_0, fire_1, ..., fire_{n-1}, NOOP]
        empty = -1
        '''

        # create all the agent nodes
        agent_nodes = []

        for agent_num in range(self.num_agents):
            agent = self.domain.agents[agent_num]
            loc_num = agent.frame.loc_num
            loc_x = self.domain_settings.AGENT_LOCATIONS[loc_num][0]
            loc_y = self.domain_settings.AGENT_LOCATIONS[loc_num][1]
            fire_reduction = agent.frame.fire_reduction

            suppressant_level = -1
            if know_all_suppressants or agent_idx == -1: # agent_idx == -1 implies state_or_obs is a full state
                suppressant_level = state_or_obs[self.num_fires + agent_num]
            elif agent_idx == agent_num:
                # this must be an obs, else agent_idx == -1
                suppressant_level = state_or_obs[self.num_fires]

            agent_node = [AGENT_NODE_NUMBER, loc_x, loc_y, fire_reduction, suppressant_level]
            agent_nodes.append(agent_node)

        # create all of the task and action nodes
        task_nodes = []
        action_nodes = []
        task_fire_nums = []

        for fire_num in range(self.num_fires):
            intensity = state_or_obs[fire_num]

            # is this fire an active task?
            if intensity > 0 and intensity < self.domain_settings.FIRE_STATES - 1:
                loc_x = self.domain_settings.FIRE_LOCATIONS[fire_num][0]
                loc_y = self.domain_settings.FIRE_LOCATIONS[fire_num][1]

                # create the task node
                task_node = [TASK_NODE_NUMBER, loc_x, loc_y, intensity, -1]
                task_nodes.append(task_node)
                task_fire_nums.append(fire_num)

                # create the action node
                action_node = [ACTION_NODE_NUMBER, fire_num, -1, -1, -1]
                action_nodes.append(action_node)

        # how many active tasks per agent?
        active_tasks_per_agent = []
        need_noop_task = False
        for agent_num in range(self.num_agents):
            num_active_tasks = 0

            agent = self.domain.agents[agent_num]
            loc_num = agent.frame.loc_num

            for fire_num in task_fire_nums:
                # can this agent fight this fire?
                if self.domain.can_perform(fire_num, loc_num):  # fire_num is location
                    num_active_tasks += 1

            active_tasks_per_agent.append(num_active_tasks)
            if num_active_tasks == 0:
                need_noop_task = True

        # do we need a NOOP task?
        if need_noop_task:
            # create a NOOP task node since someone has no active tasks for at least one agent
            task_node = [TASK_NODE_NUMBER, -1, -1, -1, -1]
            task_nodes.append(task_node)
            task_fire_nums.append(None)

        # create the NOOP action node, which can always be taken
        action_node = [ACTION_NODE_NUMBER, self.num_fires, -1, -1, -1]
        action_nodes.append(action_node)

        # determine the hyperedges connecting agents, tasks, and actions
        hyperedges = {}

        task_node_start_index = len(agent_nodes)
        action_node_start_index = task_node_start_index + len(task_nodes)
        edge_node_start_index = action_node_start_index + len(action_nodes)
        edge_num = 0

        for agent_num in range(self.num_agents):
            # get the necessary information about this agent
            agent = self.domain.agents[agent_num]
            loc_num = agent.frame.loc_num

            suppressant_level = -1
            if know_all_suppressants or agent_idx == -1: # agent_idx == -1 implies state_or_obs is a full state
                suppressant_level = state_or_obs[self.num_fires + agent_num]
            elif agent_idx == agent_num:
                # this must be an obs, else agent_idx == -1
                suppressant_level = state_or_obs[self.num_fires]

            # connect the agent to tasks and actions
            for action_node_num, action_node in enumerate(action_nodes):
                # is this a fire?
                action_num = action_node[1]

                if action_num < self.num_fires:
                    # can this agent fight this fire?
                    if self.domain.can_perform(action_num, loc_num): # action_num is fire_num is location
                        # does this agent have suppressant to act (assume yes if unknown)
                        if suppressant_level != 0:
                            # create this hyperedge
                            agent_index = agent_num
                            task_index = task_node_start_index + action_node_num  # task and action nodes are in the same order
                            action_index = action_node_start_index + action_node_num
                            edge_index = edge_node_start_index + edge_num

                            hyperedge = [agent_index, task_index, action_index]
                            hyperedges[edge_index] = hyperedge

                            edge_num += 1
                else:
                    # this is the noop action, which agents can always take for all tasks available to it
                    for task_num in range(len(task_nodes)):
                        fire_num = task_fire_nums[task_num]

                        # can this agent fight this fire?
                        if fire_num is not None and self.domain.can_perform(fire_num, loc_num): # fire_num is location
                            agent_index = agent_num
                            task_index = task_node_start_index + task_num
                            action_index = edge_node_start_index - 1  # NOOP is always the last action
                            edge_index = edge_node_start_index + edge_num

                            hyperedge = [agent_index, task_index, action_index]
                            hyperedges[edge_index] = hyperedge

                            edge_num += 1
                        elif fire_num is None and active_tasks_per_agent[agent_num] == 0:
                            # agents only have the NOOP task if they have no active fires
                            agent_index = agent_num
                            task_index = task_node_start_index + task_num
                            action_index = edge_node_start_index - 1  # NOOP is always the last action
                            edge_index = edge_node_start_index + edge_num

                            hyperedge = [agent_index, task_index, action_index]
                            hyperedges[edge_index] = hyperedge

                            edge_num += 1


        # create the edge nodes
        edge_nodes = []
        for edge_num in range(len(hyperedges)):
            node = [0] + [random.uniform(1e-5, 1e-4) for _ in range(4)]
            edge_nodes.append(node)

        # create the graph information
        feature_matrix = np.vstack(agent_nodes + task_nodes + action_nodes + edge_nodes)
        num_nodes = feature_matrix.shape[0]

        # create the adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge_index in hyperedges:
            hyperedge = hyperedges[edge_index]
            agent_index = hyperedge[0]
            task_index = hyperedge[1]
            action_index = hyperedge[2]

            adj_matrix[edge_index, agent_index] = 1.0
            adj_matrix[edge_index, task_index] = 1.0
            adj_matrix[edge_index, action_index] = 1.0

        # create the graph
        edge_index = np.where(adj_matrix > 0)
        edge_index = torch.tensor(edge_index)

        x = torch.from_numpy(feature_matrix).float()
        # x.requires_grad = True

        graph = Data(x=x, edge_index=edge_index)

        action_nodes_dict = {}
        for action_num in range(len(action_nodes)):
            action_nodes_dict[action_num + action_node_start_index] = action_nodes[action_num]

        return graph, feature_matrix, hyperedges, action_nodes_dict


    def generateGraph(self, observations, know_all_suppressants=False):
        graph_list = []
        edge_nodes_list = []
        action_space_list = []
        node_set_list = []

        for idx, obs in enumerate(observations):
            # get the graph and other information
            graph, feature_matrix, hyperedges, action_nodes_dict = self.create_graph_base(obs, idx,
                                                                                          know_all_suppressants)
            graph_list.append(graph)

            # create the action space and self edge list
            self_edge_list = []
            action_space = []
            action_indices = set()

            edge_indices = sorted(list(hyperedges.keys()))
            for edge_index in edge_indices:
                hyperedge = hyperedges[edge_index]
                agent_index = hyperedge[0]

                if agent_index == idx:
                    self_edge_list.append(edge_index)

                    action_index = hyperedge[2]
                    if action_index not in action_indices: # needed for actions that are connected to multiple tasks
                        action_indices.add(action_index)

                        action_node = action_nodes_dict[action_index]
                        action_space.append(action_node)

            edge_nodes_list.append(self_edge_list)
            action_space_list.append(action_space)

            # save our self edges
            self_edges = {}
            for edge_index in self_edge_list:
                hyperedge = hyperedges[edge_index]
                agent_index = hyperedge[0]
                task_index = hyperedge[1]
                action_index = hyperedge[2]

                array_hyperedge = [feature_matrix[agent_index], feature_matrix[task_index], feature_matrix[action_index]]
                self_edges[edge_index] = array_hyperedge
            node_set_list.append(self_edges)

        return graph_list, edge_nodes_list, action_space_list, node_set_list


    def generateCriticGraph(self, state, know_all_suppressants=False):
        graph, _, _, _ = self.create_graph_base(state, -1, know_all_suppressants)
        return graph


def test():
    # setup the parameters
    run = 1
    setup = 100
    filename_postfix = "LowestFireNum"
    training_mode = False
    steps = 20

    # create the gym object
    env = WildfireGymEnvironment(run, setup, filename_postfix, training_mode)

    # create the first state
    state = env.reset()
    observations = env.getObsFromState(state)
    print("Action Space:\n", env.action_space)
    print("Initial state:", state)
    print("Local observations:", observations)

    # create the graphs
    critic_graph = env.generateCriticGraph(state)
    print("\n##### Critic #####")
    print(critic_graph.x)
    print(critic_graph.edge_index)
    print()

    graph_list, edge_nodes_list, action_space_list, node_set_list = env.generateGraph(observations)
    for agent_num in range(env.num_agents):
        print(f"##### Agent {agent_num} #####")
        print(graph_list[agent_num].x)
        print(graph_list[agent_num].edge_index)
        print()
        print("Edge nodes:", edge_nodes_list[agent_num])
        print("Action space:", action_space_list[agent_num])
        print("Node set:", node_set_list[agent_num])
        print()


    # run the environment for 20 steps taking actions (determined by filename_postfix)
    done = False
    step = 0
    while not done and step < steps:
        # have everyone take an action
        if filename_postfix == "NOOP":
            # have each agent perform a NOOP
            actions = [env.num_fires for i in range(env.num_agents)]
        elif filename_postfix == "Random":
            actions = []

            # choose a random action for each agent
            for agent_i in range(env.num_agents):
                # does the agent have suppressant?
                if state[env.num_fires + agent_i] > 0:
                    available_actions = env.action_space[agent_i]
                    action = available_actions[np.random.choice(len(available_actions))]
                    actions.append(action)
                else:
                    # they must NOOP
                    actions.append(env.num_fires)
        elif filename_postfix == "LowestFireNum":
            actions = []

            # have each agent fight the lowest numbered fire
            for agent_i in range(env.num_agents):
                # does the agent have suppressant?
                if state[env.num_fires + agent_i] > 0:
                    available_actions = env.action_space[agent_i]
                    for action in available_actions:
                        # is this an active fire?
                        if action < env.num_fires and state[action] > 0 and state[action] < 4:
                            break
                    actions.append(action)
                else:
                    # they must NOOP
                    actions.append(env.num_fires)
        print("Actions:", actions)

        # perform an environment step
        next_state, rewards, info = env.step(actions)

        # split the new state into local observations for the different agents
        observations = env.getObsFromState(next_state)
        print("Next state:", next_state)
        print("Local observations:", observations)

        # create the graphs
        critic_graph = env.generateCriticGraph(next_state)
        print("\n##### Critic #####")
        print(critic_graph.x)
        print(critic_graph.edge_index)
        print()

        graph_list, edge_nodes_list, action_space_list, node_set_list = env.generateGraph(observations)
        for agent_num in range(env.num_agents):
            print(f"##### Agent {agent_num} #####")
            print(graph_list[agent_num].x)
            print(graph_list[agent_num].edge_index)
            print()
            print("Edge nodes:", edge_nodes_list[agent_num])
            print("Action space:", action_space_list[agent_num])
            print("Node set:", node_set_list[agent_num])
            print()

        # save the next state as the current state
        state = next_state
        step += 1


if __name__ == "__main__":
    test()
