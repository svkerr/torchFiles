-- simple-sequencer-network.lua
require 'rnn'

-- hyperparameters
batchSize = 8
rho = 5  -- sequence length
hiddenSize = 10
nIndex = 100
lr = 0.1

-- build simple recurrent network
r = nn.Recurrent(
  hiddenSize, nn.LookupTable(nIndex, hiddenSize),
  nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
  rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, nIndex))
rnn:add(nn.LogSoftMax())

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance
rnn = nn.Sequencer(rnn)

-- build criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- build dummy dataset (task is to predict next item, given previous)
sequence_ = torch.LongTensor():range(1,10)  -- [1,2,3,4,5,6,7,8,9,10]
sequence = torch.LongTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

offsets = {}
for i=1, batchSize do
  table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

-- training
iteration = 1
while iteration < 20 do
  -- 1. create a sequence of rho time-steps of input/target tables 
  inputs, targets = {}, {}
  for step=1, rho do
    -- a batch of inputs
    inputs[step] = sequence:index(1, offsets)
    -- increment indices
    offsets:add(1)
    for j=1, batchSize do
      if offsets[j] > nIndex then
        offsets[j] = 1
      end
    end
    targets[step] = sequence:index(1, offsets)
  end

  -- 2. forward sequence through rnn

  rnn:zeroGradParameters()  --zero accumulation of internal gradient parameters (rnn.forget() not required)

  -- because wrapped rnn in sequencer(), loops are not required as we feed entire tables to rnn and criterion
  outputs = rnn:forward(inputs)
  err = criterion:forward(outputs, targets)
  
  print(inputs)
  print(string.format('Iteration %d ; NLL err = %f', iteration, err))

  -- 3. backward sequence through rnn (i.e., BPTT)
  gradOutputs = criterion:backward(outputs, targets)
  gradInputs = rnn:backward(inputs, gradOutputs)
 
  -- 4. Update
  rnn:updateParameters(lr)

  iteration = iteration + 1

end

  
