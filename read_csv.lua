-- read data from CSV file into tensor
local csvFile = io.open('test.csv','r')
local header = csvFile:read()

local data = torch.Tensor(2, 3)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
    for key,val in ipairs(l) do
      data[i][key] = val
    end
end
print(data)
print(data:size())

