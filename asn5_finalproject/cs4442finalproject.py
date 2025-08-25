# ! pip install kaggle -q
# ! mkdir ~/.kaggle
# ! cp kaggle.json ~/.kaggle/
# ! kaggle datasets download arevel/chess-games
# ! unzip -qq /content/chess-games.zip
# ! pip install chess -q



import re
import numpy as np
import pandas as pd
# Garbage Collector - use it like gc.collect()
import gc
# import required module
import chess
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm


from google.colab import drive
drive.mount('/content/drive')


MAXIMUM_DATA_SIZE = 40_000





letter_2_num = {'a': 0,'b': 1,'c': 2,'d': 3,'e': 4,'f': 5,'g': 6,'h': 7}
num_2_letter = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h'}




def create_rep_layer(board, piece):
  pattern = f'[^{piece}{piece.upper()}\s]'
  s = str(board)
  # replace other pieces to .
  s = re.sub(pattern, '.', s)
  # re.sub(r'[^pP\s]', '.', s)


  # convert lower to -1, uppper to 1, represent balck and white
  s = re.sub(f'{piece}', '-1', s)
  s = re.sub(f'{piece.upper()}', '1', s)
  # replece dots to 0
  s = re.sub(f'\.', '0', s)

  board_mat = []
  # loop thourgh line by line
  for row in s.split('\n'):
    # split by white spaces
    row = row.split(' ')
    # replece string to actual integer
    row = [int(x) for x in row]
    board_mat.append(row)
  # convert to np array
  return np.array(board_mat)







def board_2_rep(board):
  pieces = ['p','r','n','b','q','k']
  layers = []
  for piece in pieces:
    layers.append(create_rep_layer(board, piece))
  board_rep = np.stack(layers)
  return board_rep



def move_2_rep(move, board):
  # convert dataset movement into uci format, 4 digit, colums rows start and end
  # d4e5. d4 -> e5
  board.push_san(move).uci()
  move = str(board.pop())

  from_output_layer = np.zeros((8,8))
  from_row = 8 - int(move[1])
  from_column = letter_2_num[move[0]]
  from_output_layer[from_row,from_column] = 1

  to_output_layer = np.zeros((8,8))
  to_row = 8 - int(move[3])
  to_column = letter_2_num[move[2]]
  to_output_layer[to_row, to_column] = 1

  return np.stack([from_output_layer, to_output_layer])



def create_move_list(s):
  return re.sub('\d*\. ', '',s).split(' ')[:-1]






# chess data loading from csv

chess_data_raw = pd.read_csv('/content/chess_games.csv', usecols=['AN', 'WhiteElo'])
# filter data less than ELO 2000
chess_data = chess_data_raw[chess_data_raw['WhiteElo'] > 2000]
# clear ram of the chess_data_raw
del chess_data_raw
gc.collect()
# chess data movement
chess_data = chess_data[['AN']]
# data cleaning
# remove game that contrain strange character
chess_data = chess_data[~chess_data['AN'].str.contains('{')]
# remove game is too short
chess_data = chess_data[chess_data['AN'].str.len() > 20]

# chess_data = chess_data[:10]
# print(chess_data.shape)
# print(chess_data.shape[0])
# (883376, 1)
# 883376




selected_size = MAXIMUM_DATA_SIZE
chess_data = chess_data[:selected_size]




# pytorch dataset dataloader

class ChessDataset(Dataset):

  def __init__(self, games):
    super(ChessDataset, self).__init__()
    self.games = games

  # dataset sample 40000 random moves before end of the sample
  def __len__(self):
    return self.games.shape[0]
    # return 2

  # get random game,moves
  def __getitem__(self, index):
    # random game
    game_i = np.random.randint(self.games.shape[0])
    random_game = chess_data['AN'].values[game_i]
    moves = create_move_list(random_game)
    # random moves from that game
    game_state_i = np.random.randint(len(moves) - 1)
    next_move = moves[game_state_i]
    moves = moves[:game_state_i]
    board = chess.Board()
    for move in moves:
      board.push_san(move)
    # convert board info and next_move to representation matrix
    x = board_2_rep(board)
    y = move_2_rep(next_move, board)
    # if move index is even, black turn
    if game_state_i % 2 == 1:
      x *= -1
    return x,y





# pytorch Dataset
data_train = ChessDataset(chess_data['AN'])
# data_train_loader = DataLoader(data_train, batch_size = 32, shuffle = False, drop_last=True)
# print(data_train.__len__())
# data_train_loader = DataLoader(data_train, batch_size = 2, shuffle = True, drop_last=True)







def display_data_loader(data_train_loader):
  x, y = next(iter(data_train_loader))

  print(f"Feature batch shape: {x.size()}")
  print(f"Labels batch shape: {y.size()}")
  x0 = x[0].squeeze()
  y0 = y[0]

  print(f"Board: {x0}")
  print(f"next_move: {y0}")

  # plt.imshow(img, cmap="gray")
  # plt.show()
  # print(f"Label: {label}")

# display_data_loader(data_train_loader)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)






class module(nn.Module):
  def __init__(self, hidden_size):
    super(module, self).__init__()
    self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(hidden_size)
    self.bn2 = nn.BatchNorm2d(hidden_size)
    self.activation1 = nn.SELU()
    self.activation2 = nn.SELU()
  def forward(self, x):
    x_input = torch.clone(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activation1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + x_input
    x = self.activation2(x)
    return x





class ChessNet(nn.Module):
  def __init__(self, hidden_layers=4, hidden_size=200):
    super(ChessNet, self).__init__()
    self.hidden_layers = hidden_layers
    self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
    self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
    self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)
  def forward(self, x):
    x = self.input_layer(x)
    x = F.relu(x)

    for i in range(self.hidden_layers):
      x = self.module_list[i](x)
    x = self.output_layer(x)

    return x


def train(network, training_set, optimizer, loss__from_function, loss__to_function, epoch = 2, batch_size = 32, ):
  """
  This function optimizes the convnet weights
  """
  #  creating list to hold loss per batch
  loss_per_batch = []

  #  defining dataloader
  train_loader = DataLoader(training_set, batch_size , shuffle = False, drop_last=True)
  # train_loader = DataLoader(training_set, batch_size , shuffle = True, drop_last=True)

  #  iterating through batches
  print('training...')

  for epoch in range(epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      # print(i)

      inputs, y = data

      inputs, y = inputs.to(device), y.to(device)

      optimizer.zero_grad()

      # Convert input data to float
      inputs = inputs.float()
      outputs = network(inputs)


      loss_from = metric_from(outputs[:,0,:], y[:,0,:])
      loss_to = metric_to(outputs[:,1,:], y[:,1,:])
      loss = loss_from + loss_to
      # loss = loss_function(classifications, labels)
      loss_per_batch.append(loss.item())


      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()

      print(f'[{epoch}] loss: {running_loss}')

      if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

  print('all done!')

  return loss_per_batch




# net = ChessNet()
net = ChessNet().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()


#  training/optimizing parameters
training_losses = train(network=net, training_set=data_train,
                        optimizer=optimizer,
                        loss__from_function=nn.CrossEntropyLoss(),
                        loss__to_function=nn.CrossEntropyLoss(),
                        epoch = 2,
                        batch_size=32)

print(training_losses)




torch.save(net, 'Chess_CNN.pth')



def predict(model, input_data, device):
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Move input data to the specified device
        input_data = input_data.to(device)
        # Make predictions using the model
        predictions = model(input_data)
    return predictions




def check_mate_single(board):
  board = board.copy()
  legal_moves = list(board.legal_moves)
  for move in legal_moves:
    board.push_uci(str(move))
    if board.is_checkmate():
      move = board.pop()
      return move
    _ = board.pop()





def distribution_over_moves(vals):
  probs = np.array(vals)
  probs = np.exp(probs)
  probs = probs / probs.sum()
  probs = probs ** 3
  probs = probs / probs.sum()
  return probs




def choose_move(board, player, color, model, device):
  legal_moves = list(board.legal_moves)

  move = check_mate_single(board)
  if move is not None:
    return move
  x = torch.Tensor(board_2_rep(board)).float().to(device)
  if color == chess.BLACK:
    x *= -1
  x = x.unsqueeze(0)
  move = predict(model, x, device)

  # print(move.shape)
  # print(move)


  vals = []
  froms = [str(legal_move)[:2] for legal_move in legal_moves]
  froms = list(set(froms))

  print(froms)

  for from_ in froms:
    val = move[0,:,:][0][8-int(from_[1]), letter_2_num[from_[0]]]
    vals.append(val)

  # print(vals)
  probs = distribution_over_moves(vals)
  choosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

  # print(choosen_from)

  vals = []
  for legal_move in legal_moves:
    from_ = str(legal_move)[:2]
    if from_ == choosen_from:
      to = str(legal_move)[2:]
      val = move[0,:,:][1][8 - int(to[1]), letter_2_num[to[0]]]
      vals.append(val)
    else:
      vals.append(0)

  choosen_move = legal_moves[np.argmax(vals)]

  return choosen_move

  pass



# Define the board state
board = chess.Board()
print(board)
# Define the current player (e.g., 'white' or 'black')
player = 'white'
# Define the color of the current player's pieces (e.g., chess.WHITE or chess.BLACK)
color = chess.WHITE
# Assuming `choose_move` is your function
move = choose_move(board, player, color, net, device)

print("Chosen move:", move)




model = ChessNet()
model = torch.load('Chess_CNN.pth')
model.eval()




























