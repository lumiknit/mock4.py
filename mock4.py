import numpy as np

class Mock4:
  def __init__(self, other=None):
    self.w = 7
    self.h = 6
    if other is None:
      self.player = 1
      self.history = []
      self.board = [0] * (self. w * self.h)
    else:
      self.player = other.player
      self.history = [x for x in other.history]
      self.board = [x for x in other.board]

  def __str__(self):
    s = ""
    s += "[ Turn {:3d} ; {}P ]\n".format(len(self.history), self.player)
    s += "|"
    for i in range(self.w):
      s += " {}".format(i)
    s += " |"
    for r in range(self.h):
      s += "\n|"
      for c in range(self.w):
        s += " {}".format(['.', 'O', 'X'][self.board[c * self.h + (self.h - r - 1)]])
      s += " |"
    return s
  
  def place(self, idx):
    z = self.h * idx
    for i in range(self.h):
      if self.board[z + i] == 0:
        self.board[z + i] = self.player
        self.player = 3 - self.player
        self.history.append(z + i)
        return z + i
    return None

  def undo(self):
    if len(self.history) > 0:
      last_pos = self.history[-1]
      self.board[last_pos] = 0
      del self.history[-1]
      return last_pos

  def check_win(self):
    # Vertical
    for i in range(self.w):
      c, last = 0, 0
      for j in range(self.h):
        b = self.board[i * self.h + j]
        if last == b:
          c += 1
          if c >= 4 and last != 0: return last
        else: c, last = 1, b
    # Horizontal
    for j in range(self.h):
      c, last = 0, 0
      for i in range(self.w):
        b = self.board[i * self.h + j]
        if last == b:
          c += 1
          if c >= 4 and last != 0: return last
        else: c, last = 1, b
    # Diagonal
    for i in range(self.w - 3):
      for j in range(self.h - 3):
        c, last = 0, 0
        for k in range(4):
          b = self.board[(i + k) * self.h + (j + k)]
          if last == b:
            c += 1
            if c >= 4 and last != 0: return last
          else: c, last = 1, b
        c, last = 0, 0
        for k in range(4):
          b = self.board[(i + 3 - k) * self.h + (j + k)]
          if last == b:
            c += 1
            if c >= 4 and last != 0: return last
          else: c, last = 1, b
    # Check Draw
    if len(self.history) >= self.w * self.h: return 0
    return None
  
  def play(self, agent1=None, agent2=None, rand_first=True, p_msg=True, p_res=True):
    def agent_user(game):
      while True:
        try:
          i = int(input("> "))
          if i < 0 or i >= game.w: raise Exception
          for j in range(game.h):
            if self.board[i * self.h + j] == 0: return i
        except: pass
    if agent1 is None: agent1 = agent_user
    if agent2 is None: agent2 = agent_user
    order = np.random.randint(2)
    if order == 0: agents = [None, agent1, agent2]
    else: agents = [None, agent2, agent1]
    if p_msg:
      print("* {} First.".format(agents[1]))
    while True:
      result = self.check_win()
      if result is not None: break
      if p_msg: print(self)
      i = agents[self.player](self)
      if self.place(i) is None:
        print("Cannot place at {}".format(i))
        return None
    if p_res:
      print("-----------------")
      print(self)
      if result == 0: print("Draw")
      else: print("{}P Win ({})".format(result, agents[result]))
    if result != 0: result = 1 + (result + order + 1) % 2
    return result

  def tensor(self, player=None):
    import torch
    if player is None: player = self.player
    imap = [0, 1, 2] if player == 1 else [0, 2, 1]
    z = torch.zeros(3, self.w, self.h, dtype=torch.float)
    for i in range(self.w * self.h):
      z[imap[self.board[i]], int(i / self.h), (i % self.h)] = 1
    return z

  def tensor_left(self):
    import torch
    z = torch.zeros(self.w, dtype=torch.long)
    for i in range(self.w):
      for j in range(self.h):
        if self.board[i * self.h + j] == 0:
          z[i] = self.h - j
          break
    return z
  
  def tensor_empty(self):
    import torch
    z = torch.zeros(self.w, dtype=torch.bool)
    for i in range(self.w):
      if self.board[(i + 1) * self.h - 1] == 0:
        z[i] = True
    return z

  def tensor_full(self):
    import torch
    z = torch.ones(self.w, dtype=torch.bool)
    for i in range(self.w):
      if self.board[(i + 1) * self.h - 1] == 0:
        z[i] = False
    return z

  def plot(self):
    import matplotlib.pyplot as plt
    fonts = [
      {'weight': 'bold', 'size': 20, 'color': 'white'},
      {'weight': 'bold', 'size': 20, 'color': 'black'},
    ]
    colors = ['black', 'white']
    plt.figure(figsize=(self.w, self.h))
    plt.axis('off')
    for i in range(1, self.h + 1):
      plt.plot([1, self.w], [i, i], zorder=1.0, c = '0.5')
    for i in range(1, self.w + 1):
      plt.plot([i, i], [1, self.h], zorder=1.0, c = '0.5')
    for i, p in enumerate(self.history):
      x = 1 + p // self.h
      y = 1 + p % self.h
      lbl = str(i + 1)
      fnt = fonts[i % 2]
      plt.gcf().gca().add_patch(
          plt.Circle(
            (x, y), 0.46, zorder=2.0,
            color=colors[i % 2], fill=True))
      plt.gcf().gca().add_patch(
          plt.Circle(
            (x, y), 0.46, zorder=2.0,
            color='black', fill=False))
      plt.text(
          x, y, lbl, zorder=3.0,
          ha='center', va='center', fontdict=fnt)

def agent_random(game):
  a = []
  for i in range(game.w):
    if game.board[(i + 1) * game.h - 1] == 0:
      a.append(i)
  if len(a) == 0: return None
  return a[np.random.randint(len(a))]

def policy_greedy_connect(game):
  def read(r, c, dr, dc):
    a = [3, 0, 0, 3, 0]
    r, c = r + dr, c + dc
    while 0 <= r and 0 <= c and r < game.h and c < game.w:
      b = game.board[c * game.h + r]
      if a[0] == 3: a[0] = b
      if b == 0 or b != a[0]: break
      a[1] += 1
      r, c = r + dr, c + dc
    while 0 <= r and 0 <= c and r < game.h and c < game.w:
      b = game.board[c * game.h + r]
      if b != 0: break
      a[2] += 1
      r, c = r + dr, c + dc
    while 0 <= r and 0 <= c and r < game.h and c < game.w:
      b = game.board[c * game.h + r]
      if a[3] == 3: a[3] = b
      if b == 0 or b != a[3]: break
      a[4] += 1
      r, c = r + dr, c + dc
    return a
  def merge(color, l, r):
    b = [0, 0, 1, 0, 0]
    if l[0] == color or l[0] == 0:
      b[2] += l[1]
      b[1] += l[2]
      if l[3] == color: b[0] += l[4]
    if r[0] == color or r[0] == 0:
      b[2] += r[1]
      b[3] += r[2]
      if r[3] == color: b[4] += r[4]
    return b
  def left(b):
    l = 4
    if b[1] + b[2] + b[3] >= 4: l = min(l, 4 - b[2])
    if b[0] + b[1] + b[2] + b[3] >= 4: l = min(l, b[1] + max(0, 4 - (b[0] + b[1] + b[2])))
    if b[1] + b[2] + b[3] + b[4] >= 4: l = min(l, b[3] + max(0, 4 - (b[2] + b[3] + b[4])))
    return l
  def value(l):
    if l < 0 or l >= 4: return 0
    return [0x1000, 0x100, 0x10, 0x1][l]
  drs = [1, 0, 1, 1]
  dcs = [1, 1, 0, -1]
  maxc, maxv = None, -100
  score = np.zeros(game.w)
  for c in range(game.w):
    acc = 0
    if game.board[(1 + c) * game.h - 1] == 0:
      r = 0
      while r < game.h and game.board[c * game.h + r] != 0: r += 1
      for d in range(4):
        la = read(r, c, drs[d], dcs[d])
        ra = read(r, c, -drs[d], -dcs[d])
        for color in range(1, 3):
          b = merge(color, la, ra)
          x = left(b)
          v = value(x)
          acc += v
      acc += 2 - abs(c - game.w // 2) * 0.1
      acc += np.random.uniform() * 0.1
      score[c] = acc / 0x05000
      if acc > maxv:
        maxv = acc
        maxc = c
  return score

def agent_greedy(game):
  score = policy_greedy_connect(game)
  return np.argmax(score)

def test_mock4(n_game, agent1, agent2):
  w1, w2 = 0, 0 
  print("** Test")
  print("* A1 = {}".format(agent1))
  print("* A2 = {}".format(agent2))
  for gi in range(n_game):
    game = Mock4()
    result = game.play(agent1, agent2, p_msg=False, p_res=False)
    if result == 1: w1 += 1
    elif result == 2: w2 += 1
  print("Total = {} games".format(n_game))
  print("W1 {} ({:.3f}) / Dr {} ({:.3f}) / W2 {} ({:.3f})".format(
      w1, w1 / n_game,
      (n_game - w1 - w2), (n_game - w1 - w2) / n_game,
      w2, w2 / n_game
  ))
