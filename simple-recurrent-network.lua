require 'rnn'

-- hyper-parameters 
batchSize = 8
rho = 5 -- sequence length
hiddenSize = 10
nIndex = 100
lr = 0.1

-- build simple recurrent neural network
r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, nIndex))
rnn:add(nn.LogSoftMax())

-- wrap the non-recurrent module (Sequential) in Recursor.
-- This makes it a recurrent module
-- i.e. Recursor is an AbstractRecurrent instance
rnn = nn.Recursor(rnn, rho)

-- build criterion

criterion = nn.ClassNLLCriterion()

-- build dummy dataset (task is to predict next item, given previous)
sequence_ = torch.LongTensor():range(1,10) -- 1,2,3,4,5,6,7,8,9,10
sequence = torch.LongTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

offsets = {}
for i=1, batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

-- training
local iteration = 1

while iteration < 500 do  -- limit number of iterations
   -- 1. create a sequence of rho time-steps
   
   local inputs, targets = {}, {}
   for step=1, rho do
      -- a batch of inputs
      inputs[step] = sequence:index(1, offsets)
      -- increment indices so that targets are +1 the input
      offsets:add(1)
      for j=1, batchSize do
         if offsets[j] > nIndex then
            offsets[j] = 1
         end
      end
      targets[step] = sequence:index(1, offsets)
   end
   
   -- 2. forward sequence through rnn
   
   rnn:zeroGradParameters()  -- zero the accumulation of the internal gradient buffers (updated parameters are unchanged)
   rnn:forget() -- forget all past time-steps
   
   local outputs, err = {}, 0
   for step=1, rho do
      outputs[step] = rnn:forward(inputs[step])
      err = err + criterion:forward(outputs[step], targets[step])
   end
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))

   -- 3. backward sequence through rnn (i.e. backprop through time)
   
   local gradOutputs, gradInputs = {}, {}
   for step=rho, 1, -1 do -- reverse order of forward calls
      gradOutputs[step] = criterion:backward(outputs[step], targets[step])
      gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
   end
   
   -- 4. update
   
   rnn:updateParameters(lr)
   
   iteration = iteration + 1
--   print(inputs[1])
end
test_input = torch.LongTensor():range(1,8)
test_out = rnn:forward(test_input)
print(inputs)

