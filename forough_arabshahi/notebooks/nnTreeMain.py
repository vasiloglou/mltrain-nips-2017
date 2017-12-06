import sys
# path to latex2sympy. Converts a latex equation to sympy format
sys.path.append('/Users/Forough/Documents/bitBucket/math-knowledge-base/Codes/latex2sympy')

import copy
import json
#from process_latex import process_sympy
from sympy import *
import re
import pprint
import mxnet as mx
import numpy as np
#from tagger import readJson
#from prover import parseEquation
from itertools import count
import random
# from equationGenerator import EquationTree

################################################################################
# math vocabulary:
functionVocab = ['Equality', 'Add', 'Mul', 'Pow',
				  'sin', 'cos', 'csc', 'sec', 'tan', 'cot',
				  'asin', 'acos', 'acsc', 'asec', 'atan', 'atan2', 'acot',
				  'sinh', 'cosh', 'csch', 'sech', 'tanh', 'coth',
				  'asinh', 'acosh', 'acsch', 'asech', 'atanh', 'acoth',
				  'log', 'exp',
				  'Min', 'Max', 'root', 'sqrt', 'IdentityFunction',
				  'range', 'const', 'var']
variables = ['Symbol']
consts = ['NegativeOne', 'Pi', 'One', 'Half', 'Integer', 'Rational']# , 'NaN', 'Infinity', 'Exp1',
nums = ['Number']
# We don't need to generate a separate class for each of the variables or functions, rather:
# constExprs = [ ConstExpr(e) for e in consts]

tmp = []
tmp.extend(functionVocab)
tmp.extend(variables)
tmp.extend(consts) 

functionDictionary = {}
ctr = 1
for f in tmp:
	functionDictionary[f] = ctr
	ctr+=1

# pprint.pprint(functionDictionary)
functionOneInp = ['sin', 'cos', 'csc', 'sec', 'tan', 'cot',
				  'asin', 'acos', 'acsc', 'asec', 'atan', 'acot',
				  'sinh', 'cosh', 'csch', 'sech', 'tanh', 'coth',
				  'asinh', 'acosh', 'acsch', 'asech', 'atanh', 'acoth',
				  'exp']# , 'IdentityFunction', 'root', 'sqrt'
functionOneInpSet = set(functionOneInp)

functionTwoInp = ['Equality', 'Add', 'Mul', 'Pow', 'log']#'Min', 'Max','atan2', 'Div'
functionTwoInpSet = set(functionTwoInp)

################################################################################
# functions: #
treeCounter = count()
def buildTree(treeType, parsedEquation, num_hidden, params, emb_dimension, varDict={}):
	# TODO: handle range
	func = str(parsedEquation.func)
	func = func.split('.')[-1]
	while func[-1]=='\'' or func[-1]=='>':
		func = func[:-1]

	if func in variables:
		# root = treeType(prefix='variables', num_hidden=num_hidden, params=params, inputName=func, args=[], emb_dimension=len(functionDictionary))
		root = treeType(prefix=func, num_hidden=num_hidden, 
			            params=params, inputName=str(func), args=[],
			            emb_dimension=len(functionDictionary), nodeNumber=next(treeCounter))
	elif func in consts:
		# root = treeType(prefix='const', num_hidden=num_hidden, params=params, inputName=func, args=[], emb_dimension=len(functionDictionary))
		root = treeType(prefix=func, num_hidden=num_hidden,
		                params=params, inputName=str(func), args=[], 
		                emb_dimension=len(functionDictionary), nodeNumber=next(treeCounter))
	elif func in functionVocab:
		root = treeType(prefix=func, num_hidden=num_hidden, 
			            params=params, args=[], inputName='',
			            emb_dimension=emb_dimension, nodeNumber=next(treeCounter))
	else:
		raise ValueError('unknown function! add to function list')

	## added this Wed, Apr 19
	# if len(parsedEquation.args) == 0:
	# 	root.args.append(treeType(prefix='data', num_hidden=num_hidden, params=params, inputName='data', args=[], emb_dimension=len(functionDictionary)))
	## up to here

	# print root.func
	#children computation
	for arg in parsedEquation.args:
		# print arg
		root.args.append(buildTree(treeType=treeType, parsedEquation=arg, 
			                       num_hidden=num_hidden, params=root._params, 
			                       emb_dimension=emb_dimension))

	# print "root args:", len(root.args)
	# print "equation args:", len(parsedEquation.args)
	return root

def one_hot(index, depth):
	out = mx.ndarray.zeros(depth)
	out[index-1] = 1
	return out


################################################################################
# classes: #

class lstmTreeInpOut(mx.rnn.BaseRNNCell):

	def __init__(self, num_hidden, emb_dimension, prefix='',  params=None, args=[], inputName='', nodeNumber=-1, dropout=0.0):
		super(lstmTreeInpOut, self).__init__(prefix='lstmTreeInpOut_'+prefix+'_', params=params)
		self.args = args
		self.func = prefix
		self.num_hidden = num_hidden
		self.emb_dimension = emb_dimension
		self.inputName = inputName
		self.nodeNumber = nodeNumber
		self.dropout = dropout
		if params is not None:
			self._params._prefix = self._prefix
		
		if self.func=='Variable' or self.func=='Const':
			self._iW = self._params.get('i2h_weight')
			self._iB = self._params.get('i2h_bias')
		elif self.func in nums:
			self._iW = self._params.get('i2h_weight')
			self._h2hW = self._params.get('h2h_weight')
			self._iB = self._params.get('i2h_bias')
			self._h2hB = self._params.get('h2h_bias')
		elif self.func in functionTwoInpSet:
			if self.func == 'Equality':
				self._iW = self._params.get('i2h_weight')
				self._iB = self._params.get('i2h_bias')
			else:
				self._h1W = self._params.get('h2h_1_weight')
				self._h1B = self._params.get('h2h_1_bias')
				self._h2W = self._params.get('h2h_2_weight')
				self._h2B = self._params.get('h2h_2_bias')
		elif self.func in functionOneInpSet:
			self._h1W = self._params.get('h2h_1_weight')
			self._h1B = self._params.get('h2h_1_bias')
		else:
			raise TypeError('uknown input function type')

	def __str__(self):
		return self.func

	def __call__(self, inp, children, memoryCh):
		"""Construct symbol for one step of treeRNN.
		Parameters
		----------
		inputs : sym.Variable
		    input symbol, 2D, batch * num_units
		states : sym.Variable
		    state from previous step or begin_state().
		Returns
		-------
		output : Symbol
		    output symbol
		states : Symbol
		    state to next step of RNN.
		"""
		name = '%s_%d_' % (self._prefix, self.nodeNumber)

		if children!=None and inp!=None:
			raise ValueError("cannot have both an input and children")

		if children==None:
			assert memoryCh==None, 'both states and memory should be None'
			if not isinstance(inp, mx.symbol.Symbol):
				print "not instance:", inp
				if inp==None:
					raise AssertionError("leaf node %s does not have input" %(str(self)))
				else:
					raise TypeError("unknown type for input: %s" %(str(type(inp))))
			#leaf
			data = inp
			if self.func in nums:
				i2h = mx.symbol.FullyConnected(data=data, weight=self._iW, bias=self._iB,
		                                       num_hidden=self.num_hidden,
		                                       name='%si2h'%name)
				act1 = mx.symbol.Activation(data=i2h, act_type="tanh", name='%sact1'%name)
				h2h = mx.symbol.FullyConnected(data=act1, weight=self._h2hW, bias=self._h2hB,
			                                   num_hidden=self.num_hidden,
			                                   name='%sh2h'%name)
				dp = mx.symbol.Dropout(data=h2h, p=self.dropout)
				next_state = mx.symbol.Activation(data=dp, act_type="sigmoid", name='%sstate'%name)

				next_memory = mx.symbol.zeros(shape=(0,self.num_hidden), name='%sbegin_memory'%(name)) # consider feeding shape??

			elif self.func=='Variable' or self.func=='Const':
				i2h = mx.symbol.FullyConnected(data=data, weight=self._iW, bias=self._iB,
	                            num_hidden=self.num_hidden,
	                            name='%si2h'%name)
				dp = mx.symbol.Dropout(data=i2h, p=self.dropout)
				# state = mx.symbol.Activation(data=i2h, act_type="sigmoid", name='%sstate'%name)
				next_state = mx.symbol.Activation(data=dp, act_type="sigmoid", name='%sembed'%name)

				next_memory = mx.symbol.zeros(shape=(0,self.num_hidden), name='%sbegin_memory'%(name)) # consider feeding shape??
			else:
				raise TypeError('unknown leaf function type')


		elif children==[]:
			print "self.inputName:", self.inputName
			raise AssertionError("something weird is going on. inputName is %s and func is %s" %(str(inp), str(self)))

		elif inp==None:
			if len(children) == 0:
				raise AssertionError('child node of %s does not have input' %(str(self)))

			elif len(children)==1:
				assert len(memoryCh) == 1, "children should have the same number of memory and state states"
				
				# memory = memoryCh[0]
				# state = children[0]
				h2h_1 = mx.symbol.FullyConnected(data=children[0], weight=self._h1W, bias=self._h1B, 
					                           num_hidden=self.num_hidden*4, name='%sh2h_1'%name)

				gates = h2h_1
				gates_dp = mx.symbol.Dropout(data=gates, p=self.dropout)

				slice_gates = mx.symbol.SliceChannel(gates_dp, num_outputs=4,
	                                          name='%sslice'%name)
				in_gate = mx.symbol.Activation(slice_gates[0], act_type="sigmoid",
	                                    name='%si'%name)
				forget_gate_1 = mx.symbol.Activation(slice_gates[1], act_type="sigmoid",
	                                        name='%sf'%name)
				in_transform = mx.symbol.Activation(slice_gates[2], act_type="tanh",
				                                 name='%sc'%name)
				out_gate = mx.symbol.Activation(slice_gates[3], act_type="sigmoid",
				                             name='%so'%name)
				next_memory = mx.symbol._internal._plus(forget_gate_1 * memoryCh[0], in_gate * in_transform,
				                                name='%smemory'%name)
				next_state = mx.symbol._internal._mul(out_gate, mx.symbol.Activation(next_memory, act_type="tanh"),
				                               name='%sstate'%name)

			elif len(children)==2:
				assert len(memoryCh) == 2, "children should have the same number of memory and states"

				if self.func == 'Equality':
					# TODO: check dim=1?
					data = mx.symbol._internal._mul(children[0], children[1], dim=1)
					data = mx.symbol.sum(data)
					data = mx.sym.FullyConnected(data=data, weight=self._iW,
					                             bias=self._iB, num_hidden=1)
					next_state = mx.sym.reshape(data=data, shape=(1,), name='%sstate'%name)
					next_memory = None
					
				else:
				
					h2h_1 = mx.symbol.FullyConnected(data=children[0], weight=self._h1W, bias=self._h1B, 
						                           num_hidden=self.num_hidden*5, name='%sh2h_1'%name)
					h2h_2 = mx.symbol.FullyConnected(data=children[1], weight=self._h2W, bias=self._h2B, 
						                           num_hidden=self.num_hidden*5, name='%sh2h_2'%name)
					gates = h2h_1 + h2h_2
					gates_dp = mx.symbol.Dropout(data=gates, p=self.dropout)

					slice_gates = mx.symbol.SliceChannel(gates_dp, num_outputs=5,
		                                          name='%sslice'%name)
					in_gate = mx.symbol.Activation(slice_gates[0], act_type="sigmoid",
		                                    name='%si'%name)
					forget_gate_1 = mx.symbol.Activation(slice_gates[1], act_type="sigmoid",
		                                        name='%sf1'%name)
					forget_gate_2 = mx.symbol.Activation(slice_gates[2], act_type="sigmoid",
		                                        name='%sf2'%name)
					in_transform = mx.symbol.Activation(slice_gates[3], act_type="tanh",
					                                 name='%sc'%name)
					out_gate = mx.symbol.Activation(slice_gates[4], act_type="sigmoid",
					                             name='%so'%name)
					forget_memory = mx.symbol._internal._plus(forget_gate_1 * memoryCh[0], forget_gate_2 * memoryCh[1],
						                                name='%sfor_mem'%name)
					next_memory = mx.symbol._internal._plus(forget_memory, in_gate * in_transform,
					                                name='%smemory'%name)
					next_state = mx.symbol._internal._mul(out_gate, mx.symbol.Activation(next_memory, act_type="tanh"),
					                               name='%sstate'%name)


			elif len(children)>2:
				print "parent:", self
				print "children:", [children[i] for i in range(len(children))]
				raise ValueError("the number of children should not exceed 2")

		else: 
			raise AssertionError("nor leaf nor non-leaf!!!")


		return next_state, next_memory

	def unroll(self, dataNameDictionary):
		states_children = []
		memory_children = []

		for arg in self.args:
			states, memory = arg.unroll(dataNameDictionary=dataNameDictionary)
			states_children.append(states)
			memory_children.append(memory)

		if self.inputName=='':
			output_state, output_memory = self(inp=None, children=states_children, memoryCh=memory_children)
		elif self.inputName!='' and len(states_children)!=0:
			raise ValueError("non-leaf node has input!")

		else:
			#in leaf
			if self.inputName == '':
				raise AssertionError, "leaf does not have input name"
			inputs = dataNameDictionary[self.func+'_'+self.inputName+'_%d'%(self.nodeNumber)]
			if self.func in nums:
				output_state, output_memory = self(inp=inputs, children=None, memoryCh=None)
			else:
				inputDp = mx.symbol.Dropout(data=inputs, p=self.dropout)
				output_state, output_memory = self(inp=inputDp, children=None, memoryCh=None)

		return output_state, output_memory

	# def unrollUpToNode(self, nodeNumber, dataNameDictionary):
	# 	newNode = findNode(self, nodeNumber)

	# 	return newNode.unroll(dataNameDictionary)

	def getDataNames(self, dataNames=[], nodeNumbers=[]):
		if len(self.args)==0:
			# if self.inputName not in set(dataNames):
			dataNames.append(self.func+'_'+self.inputName)
			nodeNumbers.append(self.nodeNumber)

		for arg in self.args:
			arg.getDataNames(dataNames, nodeNumbers)
		# return list(set(dataNames))
		return [dataNames, nodeNumbers]

	def traverse(self):
		print self.func
		for arg in self.args:
			arg.traverse()

	def isNumeric(self):
		flag = False
		for arg in self.args:
			flag = flag or arg.isNumeric()

		if self.func == 'Number':
			return True
		else:
			return (False or flag)

class BucketEqIteratorInpOut(mx.io.DataIter):
	"""Simple bucketing iterator for tree LSTM model for equations.
    Label for each step is constructed from data of
    next step.
    Parameters
    ----------
    enEquations : list of list of int
        encoded equations
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
    """
	def __init__(self, enEquations, eqTreeList, batch_size, labels, vocabSize,buckets=None, invalid_label=-1,
	             label_name='softmax_label', dtype='float32',
	             layout='NTC', num_hidden=None, bound=[-3.14,3.15], devFlag=0):
		super(BucketEqIteratorInpOut, self).__init__()

		self.vocabSize = vocabSize

		buckets = np.arange(len(enEquations))

		self.data = [[] for _ in buckets]
		self.dataFlag = [[] for _ in buckets]
		self.data_name = [[] for _ in buckets]
		self.upperBound = bound[1]
		self.lowerBound = bound[0]
		self.devFlag = devFlag

		for i, eq in enumerate(enEquations):
			buck = i
			eq = eq
			buff = []
			flag = []
			for j in range(len(eq)):
				if isinstance(eq[j],list):
					print "I am a list"
					buff.append(np.array(eq[j], dtype=dtype))
				else:
					if re.search('_', str(eq[j])):
						#print 'found a number'
						num = float(eq[j].split('_')[1])
						assert num >= 0, 'numbers should all be positive. Negative numbers are modeled by x -1'
						# normalizing the input number to the max
						# print 'before norm:', num
						# should we round or not? I think we should not
						# num = round(num/self.upperBound,2)
						#num = num/self.upperBound
						num = num
						# print 'after norm:', num
						tmp = np.array([num])
						buff.append(tmp)
						flag.append(1)
					else:
						tmp = np.zeros((1,vocabSize), dtype=dtype)
						tmp[0][eq[j]] = 1.0
						buff.append(tmp)
						flag.append(0)

			# print 'flag:', flag
			assert len(buff)==len(flag)
			self.data[buck].extend(buff)
			self.dataFlag[buck].extend(flag)
			[dn, nn] = eqTreeList[i].getDataNames([],[])
			assert len(dn)==len(buff)
			dn = [dn[j]+'_%d'%(nn[j]) for j in range(len(dn))]
			self.data_name[buck].append(dn)

		self.batch_size = batch_size
		self.buckets = [bucketIndex(bucket, self.devFlag) for bucket in buckets] # buckets
		self.label_name = label_name

		self.dtype = dtype
		self.invalid_label = invalid_label
		self.nddata = []
		self.ndlabel = []
		self.major_axis = layout.find('N')
		self.labels = labels
		self.default_bucket_key = 0# max(buckets) # what is our default bucket key?

		
		if self.major_axis == 0:
		
			self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (batch_size, vocabSize)) 
			                      if self.dataFlag[self.default_bucket_key][i]==0 else
			                      (self.data_name[self.default_bucket_key][0][i], (batch_size, )) 
			                      for i in range(len(self.data_name[self.default_bucket_key][0]))]
                        #self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (batch_size, vocabSize)) 
			#                      for i in range(len(self.data_name[self.default_bucket_key][0]))]
		
			self.provide_label = [(label_name, (batch_size, ))]
		

		elif self.major_axis == 1:
		
			self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (vocabSize, batch_size))
			                      if self.dataFlag[0][i] == 0 else
			                      (self.data_name[self.default_bucket_key][0][i], (1, batch_size))
			                      for i in range(len(self.data_name[self.default_bucket_key][0]))]
		
			#self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (vocabSize, batch_size))
			#                      for i in range(len(self.data_name[self.default_bucket_key][0]))]
			self.provide_label = [(label_name, (1, batch_size))]

		else:
			raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

		self.idx = []
		for i, buck in enumerate(self.data):
			self.idx.extend([(i, 0)])
		self.curr_idx = 0

		self.reset()

	def reset(self):
		self.curr_idx = 0

		self.nddata = []
		self.ndlabel = []
		for i, buck in enumerate(self.data):

			label = self.labels[i]
			self.nddata.append([mx.ndarray.array(buck[k], dtype=self.dtype) for k in range(len(buck))])
			self.ndlabel.append(mx.ndarray.array(label, dtype=self.dtype))


	def next(self):
		if self.curr_idx == len(self.idx):
			raise StopIteration
		i, j = self.idx[self.curr_idx]
		self.curr_idx += 1

		if self.major_axis == 1:
			data = self.nddata[i].T
			label = self.ndlabel[i].T
		else:
			data = self.nddata[i]
			label = self.ndlabel[i]


		# print 'dataNames:', self.data_name[i][0]
		# print 'dataFlag:', self.dataFlag[i]
		# print 'provided data:', [(self.data_name[i][0][j], (self.batch_size, self.vocabSize))
  #                               if self.dataFlag[i][j] == 0 else
  #                               (self.data_name[i][0][j], (self.batch_size, ))
  #                               for j in range(len(self.data_name[i][0]))]
                #d = mx.io.DataBatch(data, [label], pad=0,
		#                 bucket_key=bucketIndex(self.buckets[i], self.devFlag),
		#                 provide_data=[(self.data_name[i][0][j], (self.batch_size, self.vocabSize)) 
		#                                for j in range(len(self.data_name[i][0]))],
		#                 provide_label=[(self.label_name, label.shape)])
		d = mx.io.DataBatch(data, [label], pad=0,
		                 bucket_key=self.buckets[i], #bucketIndex(self.buckets[i], self.devFlag),
		                 provide_data=[(self.data_name[i][0][j], (self.batch_size, self.vocabSize))
		                                if self.dataFlag[i][j] == 0 else
		                                (self.data_name[i][0][j], (self.batch_size, ))
		                                for j in range(len(self.data_name[i][0]))],
		                 provide_label=[(self.label_name, label.shape)])

		return d

class bucketIndex(object):
	def __init__(self, index, devFlag=0):
		self.bucketIDX = index
		self.devFlag = devFlag

class Accuracy(mx.metric.EvalMetric):
	def __init__(self, axis=1):
		super(Accuracy, self).__init__(name='accuracy')
		self.axis = axis

	def update(self, labels, preds):

		mx.metric.check_label_shapes(labels, preds)


		for label, pred in zip(labels, preds):

			if pred.shape != label.shape:
				pred = mx.ndarray.argmax(pred, axis=self.axis)
			elif pred.shape == (1,):
				pred = mx.nd.array(pred.asnumpy().round())

			pred_label = pred.asnumpy().astype('int32')
			label = label.asnumpy().astype('int32')

			self.sum_metric += (pred_label.flat == label.flat).sum()
			self.num_inst += len(pred_label.flat)
		
class recall(mx.metric.EvalMetric):
	def __init__(self, axis=1):
		super(recall, self).__init__(name='recall')
		self.axis = axis

	def update(self, labels, preds):
		
		# print 'labels:', labels
		# print 'preds:', preds
		mx.metric.check_label_shapes(labels, preds)

		true_positives, false_positives, false_negatives = 0., 0., 0.
		true_negatives = 0.

		for label, pred in zip(labels, preds):

			if pred.shape != label.shape:
				pred = mx.ndarray.argmax(pred, axis=self.axis)
			elif pred.shape == (1,):
				pred = mx.nd.array(pred.asnumpy().round())
			label = label.astype('int32')
			pred_label = pred.astype('int32').as_in_context(label.context)

			if not isinstance(pred, list):
				pred = [pred]
			if not isinstance(label, list):
				label = [label]
			if not isinstance(pred_label, list):
				pred_label = [pred_label]
			mx.metric.check_label_shapes(label, pred)
			# if len(np.unique(label)) > 2:
			# 	raise ValueError("recall currently only supports binary classification.")

			for y_pred, y_true in zip(pred_label, label):
				y_pred = y_pred.asscalar()
				y_true = y_true.asscalar()
				if y_pred == 1 and y_true == 1:
					# true positives
					self.sum_metric += 1.
				if y_true == 1:
					# true positives + false positives
					self.num_inst += 1


class precision(mx.metric.EvalMetric):
	def __init__(self, axis=1):
		super(precision, self).__init__(name='precision')
		self.axis = axis

	def update(self, labels, preds):
		
		mx.metric.check_label_shapes(labels, preds)

		true_positives, false_positives, false_negatives = 0., 0., 0.
		true_negatives = 0.

		for label, pred in zip(labels, preds):

			if pred.shape != label.shape:
				pred = mx.ndarray.argmax(pred, axis=self.axis)
			elif pred.shape == (1,):
				pred = mx.nd.array(pred.asnumpy().round())
			label = label.astype('int32')
			pred_label = pred.astype('int32').as_in_context(label.context)
			#print 'pred_label:', pred_label.asnumpy()

			if not isinstance(pred, list):
				pred = [pred]
			if not isinstance(label, list):
				label = [label]
			if not isinstance(pred_label, list):
				pred_label = [pred_label]
			mx.metric.check_label_shapes(label, pred)
			# if len(np.unique(label)) > 2:
			# 	raise ValueError("recall currently only supports binary classification.")

			for y_pred, y_true in zip(pred_label, label):
				y_pred = y_pred.asscalar()
				y_true = y_true.asscalar()
				if y_pred == 1 and y_true == 1:
					# true positives
					self.sum_metric += 1.
				if y_pred == 1:
					# true positives + false positives
					self.num_inst += 1


################################################################################
# main: #
def main():

	# pprint.pprint(functionDictionary)

	# params = mx.rnn.RNNParams()
	params = None
	contexts = mx.cpu(0)
	num_hidden = 100
	vocabSize = len(functionDictionary)
	emb_dimension = 16
	out_dimension = 32
	batch_size = 1

	inputPath = "smallTestMxnet.json"
	jsonAtts = ["variables", "CCGparse", "equation","sentence","equation","equation"]

	parseTreeList = [] # list of lists
	rawLine = [] # list of lists
	equations = [] 
	parsedEquations = []
	variables = []
	ranges = []
	parsedRanges = []
	labels = []

	#reading input and parsing input equations
	readJson(inputPath, parseTreeList, rawLine, equations, variables, ranges, labels, jsonAtts)
	parseEquation(equations, parsedEquations)
	parseEquation(ranges, parsedRanges)
	numSamples = len(parsedEquations)
	buckets = list(xrange(numSamples))
	labels = mx.nd.ones([numSamples,])
	# print equations	
	print "parsedEquations:", parsedEquations[23]
	# print "labels:", labels
	
	samples = []
	dataNames = []
	ctr = 0
	for equation in parsedEquations:
		# treeCounter = count()
		currNNTree = buildTree(treeType=nnTree , parsedEquation=equation, 
			                   num_hidden=num_hidden, params=params, 
			                   emb_dimension=emb_dimension)
		# currNNTree.traverse()

		# state = currNNTree.unroll()

		# print "traversing equation ", ctr
		# currTreeLSTM.traverse()
		# print "travesal done"
		# print state
		currDataNames = currNNTree.getDataNames(dataNames=[])
		# print "currDataNames:", currDataNames
		dataNames.append(currDataNames)
		samples.append(currNNTree)
		# ctr += 1
	# Samples are stored in samples. The data iterator is then only a list iterator. (I think)

	train_eq, _ = encode_equations(parsedEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	# data_train = mx.rnn.BucketSentenceIter(train_eq, batch_size)
	data_train  = BucketEqIterator(enEquations=train_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=labels,
                                            invalid_label=-1)

	

	# print "parse Length:", len(parsedEquations)
	# print "parsed equations:", parsedEquations
	# print "encoded equations:", train_eq
	# print dataNames

	# print "data_train:", data_train.provide_data
	# print "self index:", data_train.idx
	# print "self current index:", data_train.curr_idx
	# d = data_train.next()
	# print "self current index:", data_train.curr_idx
	# print "data_train:", d.provide_data
	# d = data_train.next()
	# print "self current index:", data_train.curr_idx
	# print "data_train:", d.provide_data
	# d = data_train.next()
	# print "self current index:", data_train.curr_idx
	# print "data_train:", d.provide_data
	# print "data_label:", d.provide_label

	# assert 1==2, "stop"

	def sym_gen(bucketIDX):
		# print "in sym_gen"
		data = mx.sym.Variable('data')
		label = mx.sym.Variable('softmax_label')
		# embed = mx.sym.Embedding(data=data, input_dim=len(functionVocab),
		#                          output_dim=args.num_embed, name='embed')

		
		# We need to figure out how to use the bucketIDX. 
		# I think the original one handles it using the data iterator.
		# We might be able to handle this using mx.rnn.BucketSentenceIter
		tree = samples[bucketIDX]
		dataNames = tree.getDataNames(dataNames=[])
		nameDict = {}
		for dn in set(dataNames):
			if dn not in nameDict:
				nameDict[dn] = mx.sym.Variable(dn)
		outputs = tree.unroll(nameDict)
		# data_names = dataNames[bucketIDX]
		# print data_names
		dataNames = list(set(dataNames))
		# data = mx.sym.Group([value for _, value in dataNames.iteritems()])


		# pred = mx.sym.Reshape(outputs, shape=(-1, tree._num_hidden))
		pred = mx.sym.FullyConnected(data=outputs, num_hidden=out_dimension, name='pred')

		label = mx.sym.Reshape(label, shape=(-1,))
		pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

		# return pred, ('data',), ('softmax_label',)
		return pred, (dataNames), ('softmax_label',)

	model = mx.mod.BucketingModule(
		sym_gen             = sym_gen,
		default_bucket_key  = 0,
		context             = contexts)

	model.fit(
        train_data          = data_train,
        eval_data           = data_train,
        eval_metric         = mx.metric.Perplexity(0),
        kvstore             = 'str',
        optimizer           = 'sgd',
        optimizer_params    = { 'learning_rate': 0.01,
                                'momentum': 0.0,
                                'wd': 0.00001 },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = 1,
        batch_end_callback  = mx.callback.Speedometer(1, 20))



	# train_eq, _ = encode_equations(parsedEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)


if __name__=='__main__':
	main()








