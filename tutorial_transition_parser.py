from __future__ import print_function
from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import re

# actions the parser can take
SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NUM_ACTIONS = 5

class Vocab:
  def __init__(self, w2i):
    self.w2i = dict(w2i)
    self.i2w = {i:w for w,i in w2i.iteritems()}
  @classmethod
  def from_list(cls, words):
    w2i = {}
    idx = 0
    for word in words:
      w2i[word] = idx
      idx += 1
    return Vocab(w2i)
  @classmethod
  def from_file(cls, vocab_fname):
    words = []
    with file(vocab_fname) as fh:
      for line in fh:
        line.strip()
        word, count = line.split()
        words.append(word)
    return Vocab.from_list(words)

  def size(self): return len(self.w2i.keys())


class AMRAction:
  def __init__(self, action, label, index):
    self.action = action
    self.label = label
    self.index = index
  def __repr__(self):
    return "action: %s label: %s index: %s" % (self.action, self.label, self.index)

  @classmethod
  def from_oracle(cls, labeled_action, va):
    split_action = labeled_action.split("_")
    action = split_action[0]
    label = None
    if len(split_action) == 2:
      label = split_action[1]
    return AMRAction(action, label, va.w2i[action])

class Node():
    def __init__(self, label, token):
        self.label = label
        self.token = token
        self.children = []

    def add_child(self, obj, relation):
        self.children.append((obj, relation))
    def preety_print(self, depth = 1):
      str = "( %s orig: %s" % ( self.label, self.token);
      str += "".join(("\n".ljust(depth+1, "\t") + "%s  %s" % (relation, child.preety_print(depth + 1))) for (child, relation) in self.children)
      if(self.children):
        str += "\n".ljust(depth, "\t")
      str += ")"
      return str;

def read_oracle(fname, vw, va):
  with file(fname) as fh:
    for line in fh:
      line = line.strip()
      ssent, sacts = re.split(r' \|\|\| ', line)
      sent = [vw.w2i[x] for x in ssent.split()]
      acts = read_actions(sacts, va)
      yield (sent, acts)

def read_actions(sacts, va):
  if('\'' in sacts):
    # actions format: ['SH_label', 'RL_label', 'RR_label', 'DN']
    actions = sacts[2:-2].split('\', \'')
  else:
    actions = sacts.split()
  parser_actions = [AMRAction.from_oracle(x, va) for x in actions]
  return parser_actions

WORD_DIM = 64
LSTM_DIM = 64
ACTION_DIM = 32

class TransitionParser:
  def __init__(self, model, vocab):
    self.vocab = vocab
    self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
    self.pb_comp = model.add_parameters((LSTM_DIM, ))
    self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
    self.pb_s2h = model.add_parameters((LSTM_DIM, ))
    self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
    self.pb_act = model.add_parameters((NUM_ACTIONS, ))

    # layers, in-dim, out-dim, model
    self.buffRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
    self.stackRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
    self.pempty_buffer_emb = model.add_parameters((LSTM_DIM,))
    nwords=vocab.size()
    self.WORDS_LOOKUP = model.add_lookup_parameters((nwords, WORD_DIM))

  # returns an expression of the loss for the sequence of actions
  # (that is, the oracle_actions if present or the predicted sequence otherwise)
  def parse(self, t, oracle_actions=None):
    dy.renew_cg()
    if oracle_actions:
      oracle_actions = list(oracle_actions)
      oracle_actions.reverse()
    stack_top = self.stackRNN.initial_state()
    toks = list(t)
    toks.reverse()
    stack = []
    cur = self.buffRNN.initial_state()
    buffer = []
    empty_buffer_emb = dy.parameter(self.pempty_buffer_emb)
    W_comp = dy.parameter(self.pW_comp)
    b_comp = dy.parameter(self.pb_comp)
    W_s2h = dy.parameter(self.pW_s2h)
    b_s2h = dy.parameter(self.pb_s2h)
    W_act = dy.parameter(self.pW_act)
    b_act = dy.parameter(self.pb_act)
    losses = []
    for tok in toks:
      tok_embedding = self.WORDS_LOOKUP[tok]
      cur = cur.add_input(tok_embedding)
      buffer.append((cur.output(), tok_embedding, self.vocab.i2w[tok]))

    while not (len(stack) == 1 and len(buffer) == 0):
      # based on parser state, get valid actions
      valid_actions = []
      if len(buffer) > 0:  # can only reduce if elements in buffer
        valid_actions += [SH]
      if len(stack) >= 1:
        valid_actions += [DN]
      if len(stack) >= 2:  # can only shift if 2 elements on stack
        valid_actions += [RL, RR]
      if len(stack) >= 3:
        valid_actions += [SW] # can only swap if we have at least 3 elements on the stack

      # compute probability of each of the actions and choose an action
      # either from the oracle or if there is no oracle, based on the model
      action = valid_actions[0]
      label = None
      log_probs = None
      if len(valid_actions) > 1:
        buffer_embedding = buffer[-1][0] if buffer else empty_buffer_emb
        stack_embedding = stack[-1][0].output() # the stack has something here
        parser_state = dy.concatenate([buffer_embedding, stack_embedding])
        h = dy.tanh(W_s2h * parser_state + b_s2h)
        logits = W_act * h + b_act
        log_probs = dy.log_softmax(logits, valid_actions)
        if oracle_actions is None:
          print('no oracle!')
          action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
      if oracle_actions is not None:
        oracle_action = oracle_actions.pop()
        action = oracle_action.index
        label = oracle_action.label
        if log_probs is not None:
          # append the action-specific loss based on oracle
          losses.append(dy.pick(log_probs, action))
      # execute the action to update the parser state
      if action == SH:
        _, tok_embedding, token = buffer.pop()
        stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
        stack_state = stack_state.add_input(tok_embedding)
        stack.append((stack_state, Node(label, token)))
      elif action == DN:
        buffer.pop()
      elif action == SW:
        top = stack.pop()
        mid = stack.pop()
        lower = stack.pop()
        stack.append(mid)
        stack.append(lower)
        stack.append(top)
      else: # one of the reduce actions
        right = stack.pop()
        left = stack.pop()
        head, modifier = (left, right) if action == RR else (right, left)
        top_stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
        head_rep, head_node = head[0].output(), head[1]
        mod_rep, mod_node = modifier[0].output(), modifier[1]
        composed_rep = dy.rectify(W_comp * dy.concatenate([head_rep, mod_rep]) + b_comp)
        top_stack_state = top_stack_state.add_input(composed_rep)
        head_node.add_child(mod_node, label)
        stack.append((top_stack_state, head_node))
        if oracle_actions is None:
          print('{0} --> {1}'.format(head_node.token, mod_node.token))

    # the head of the tree that remains at the top of the stack is now the root
    head = stack.pop()[1]
    if oracle_actions is None:
      print('ROOT --> {0}'.format(head))
    # print("losses" + str(map(lambda x: x.scalar_value(), losses)))
    # print(head.preety_print())
    return (-dy.esum(losses) if losses else None, head)

acts = ['SH', 'RL', 'RR', 'DN', 'SW']
vocab_acts = Vocab.from_list(acts)

vocab_words = Vocab.from_file('data/vocab.txt')
train = list(read_oracle('data/amr-examples.txt', vocab_words, vocab_acts))
dev = list(read_oracle('data/amr-examples-test.txt', vocab_words, vocab_acts))

model = dy.Model()
trainer = dy.AdamTrainer(model)

tp = TransitionParser(model, vocab_words)

#cmake .. -DEIGEN3_INCLUDE_DIR=/Users/flo/Documents/Doctorat/AMR/dynet-base/eigen -DBOOST_ROOT=/usr/local/opt/boost160/ -DPYTHON=/usr/bin/python
min_loss = 100
rounds = 0
min_epoch = 0
for epoch in range(100):
  for (s,a) in train:
    loss = tp.parse(s, a)[0]
    if loss is not None:
      # for some weird reason backward throws an failed assertion if there is no scalar value retrievall
      loss.scalar_value()
      loss.backward()
      trainer.update()
  dev_words = 0
  dev_loss = 0.0
  for (ds, da) in dev:
    loss = tp.parse(ds, da)[0]
    dev_words += len(ds)
    if loss is not None:
      dev_loss += loss.scalar_value()
  loss_dev_words = dev_loss / dev_words
  print('[validation] epoch {}: per-word loss: {}'.format(epoch, loss_dev_words))
  min_loss = min(min_loss, loss_dev_words)
  if min_loss == loss_dev_words:
    rounds = 0
    min_epoch = epoch
  else:
    rounds += 1
  print("since {} min loss {} for {} rounds.".format(min_epoch, min_loss, rounds))
