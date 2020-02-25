# Description:
# Single Player Monte Carlo Tree Search implementation.
# This is a Python implementation of the single player
# Monte Carlo tree search as described in the paper:
# https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf

import mcts_node as nd
import numpy as np
import os
# Import your game implementation here.
import env.crew_and_jobs as game


class MCTS:
	"""
	Class for Single Player Monte Carlo Tree Search implementation.
	"""

	def __init__(self, node, verbose=False):
		"""
		Constructor.

		Args:
			node: Root node of the tree of class Node.
			verbose: True: printdetails of search during execution; False: Otherwise
		"""
		self.root = node
		self.verbose = verbose

	def selection(self):
		"""
		Performs selection phase of the MCTS.
		:param
		:return: Selected Child Node
		"""
		selected_child = self.root
		has_child = False

		# Check if child nodes exist.
		if len(selected_child.children) > 0:
			has_child = True
		else:
			has_child = False

		while has_child:
			selected_child = self.select_child(selected_child)
			if len(selected_child.children) == 0:
				has_child = False
			# SelectedChild.visits += 1.0

		if self.verbose:
			print("\nSelected: ", game.GetStateRepresentation(selected_child.state))

		return selected_child

	def select_child(self, node):
		"""
		Given a Node, selects the first unvisited child Node, or if all
		children are visited, selects the Node with greatest UTC value.
		:param node: Node from which to select child Node from.
		:return: Selected Child Node
		"""
		if len(node.children) == 0:
			return node

		for child in node.children:
			if child.visits > 0.0:
				continue
			else:
				if self.verbose:
					print("Considered child", game.GetStateRepresentation(child.state), "UTC: inf",)
				return child

		max_weight = 0.0
		for child in node.children:
			# Weight = self.EvalUTC(Child)
			weight = child.sputc
			if self.verbose:
				print("Considered child:", game.GetStateRepresentation(child.state), "UTC:", weight)
			if weight > max_weight:
				max_weight = weight
				selected_child = child
		return selected_child

	def expansion(self, leaf):
		"""
		Performs expansion phase of the MCTS.
		:param leaf: Leaf Node to expand.
		:return:
		"""
		if self.is_terminal(leaf):
			print("Is Terminal.")
			return False
		elif leaf.visits == 0:
			return leaf
		else:
			# Expand.
			if len(leaf.children) == 0:
				children = self.eval_children(leaf)
				for new_child in children:
					if np.all(new_child.state == leaf.state):
						continue
					leaf.append_child(new_child)
			assert (len(leaf.children) > 0), "Error"
			child = self.select_child_node(leaf)

		if self.verbose:
			print("Expanded: ", game.GetStateRepresentation(child.state))
		return child

	def is_terminal(self, node):
		"""
		Checks if a Node is terminal (it has no more children).
		:param node: Node to check.
		:return:
		"""
		if game.IsTerminal(node.state):
			return True
		else:
			return False

	def eval_children(self, node):
		"""
		Evaluates all the possible children states given a Node state
		and returns the possible children Nodes.
		:param node: Node from which to evaluate children.
		:return: possible children Nodes.
		"""
		next_states = game.EvalNextStates(node.state)
		children = []
		for State in next_states:
			child_node = nd.Node(State)
			children.append(child_node)

		return children

	def select_child_node(self, node):
		"""
		Selects a child node randomly.
		:param node: Node from which to select a random child.
		:return:
		"""
		len = len(node.children)
		assert len > 0, "Incorrect length"
		i = np.random.randint(0, len)
		return node.children[i]

	def simulation(self, node):
		"""
		Performs the simulation phase of the MCTS.
		:param node: Node from which to perform simulation.
		:return:
		"""
		current_state = node.state
		if self.verbose:
			print("Begin Simulation")

		level = self.get_level(node)
		# Perform simulation.
		while not(game.IsTerminal(current_state)):
			current_state = game.GetNextState(current_state)
			level += 1.0
			if self.verbose:
				print("CurrentState:", game.GetStateRepresentation(current_state))
				game.PrintTablesScores(current_state)

		result = game.GetResult(current_state)
		return result

	def back_propagation(self, node, result):
		"""
		Performs the back propagation phase of the MCTS.
		:param node: Node from which to perform Back propagation.
		:param result: Result of the simulation performed at Node.
		:return:
		"""
		# Update Node's weight.
		current_node = node
		current_node.wins += result
		current_node.ressq += result ** 2
		current_node.visits += 1
		self.eval_utc(current_node)

		while self.has_parent(current_node):
			# Update parent node's weight.
			current_node = current_node.parent
			current_node.wins += result
			current_node.ressq += result ** 2
			current_node.visits += 1
			self.eval_utc(current_node)
		# self.root.wins += Result
		# self.root.ressq += Result**2
		# self.root.visits += 1
		# self.EvalUTC(self.root)

	def has_parent(self, node):
		"""
		Checks if Node has a parent.
		:param node: Node to check.
		:return:
		"""
		if node.parent is None:
			return False
		else:
			return True

	def eval_utc(self, node):
		"""
		Evaluates the Single Player modified UTC. See:
		https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
		:param node: Node to evaluate.
		:return:
		"""
		# c = np.sqrt(2)
		c = 0.5
		w = node.wins
		n = node.visits
		sumsq = node.ressq
		if node.parent is None:
			t = node.visits
		else:
			t = node.parent.visits

		utc = w/n + c * np.sqrt(np.log(t)/n)
		d = 10000.
		modification = np.sqrt((sumsq - n * (w/n)**2 + d)/n)
		# print"Original", utc
		# print"Mod", modification
		node.sputc = utc + modification
		return node.sputc

	def get_level(self, node):
		"""
		Gets the level of the node in the tree.
		:param node: Node to evaluate the level.
		:return:
		"""
		level = 0.0
		while node.parent:
			level += 1.0
			node = node.parent
		return level

	def print_tree(self):
		"""
		Prints the tree to file.
		:return:
		"""
		f = open('Tree.txt', 'w')
		node = self.root
		self.print_node(f, node, "", False)
		f.close()

	def print_node(self, file, node, indent, is_terminal):
		"""
		Prints the tree Node and its details to file.
		:param file: file name.
		:param node: Node to print.
		:param indent: Indent character.
		:param is_terminal: True: Node is terminal. False: Otherwise.
		:return:
		"""
		file.write(indent)
		if is_terminal:
			file.write("\-")
			indent += "  "
		else:
			file.write("|-")
			indent += "| "

		string = str(self.get_level(node)) + ") (["
		# for i in Node.state.bins: # game specific (scrap)
		# 	string += str(i) + ", "
		string += str(game.GetStateRepresentation(node.state))
		string += "], W: " + str(node.wins) + ", N: " + str(node.visits) + ", UTC: " + str(node.sputc) + ") \n"
		file.write(string)

		for Child in node.children:
			self.print_node(file, Child, indent, self.is_terminal(Child))

	def print_result(self, result):
		filename = 'Results.txt'
		if os.path.exists(filename):
			append_write = 'a'  # append if already exists
		else:
			append_write = 'w'  # make a new file if not

		f = open(filename, append_write)
		f.write(str(result) + '\n')
		f.close()

	def run(self, max_iter=5000):
		"""
		Runs the SP-MCTS.

		:param max_iter: Maximum iterations to run the search algorithm.
		:return:
		"""
		for i in range(max_iter):
			if self.verbose:
				print("\n===== Begin iteration:", i, "=====")
			X = self.selection()
			Y = self.expansion(X)
			if Y:
				result = self.simulation(Y)
				if self.verbose:
					print("Result: ", result)
				self.back_propagation(Y, result)
			else:
				result = game.GetResult(X.state)
				if self.verbose:
					print("Result: ", result)
				self.back_propagation(X, result)
			self.print_result(result)

		print("Search complete.")
		print("Iterations:", i)
