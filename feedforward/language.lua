nn = require "nn"
require "hdf5"

cmd = torch.CmdLine()
-- logistics
cmd:option('-datafile', 'language5.hdf5', 'data file')
cmd:option('-predfile', 'preds.txt', 'file to write predictions to')
cmd:option('-modelfile', 'embeds.hdf5', 'file to save model to or load from')
cmd:option('-cuda', 1, '0 if use cpu')

-- training options
cmd:option('-nepochs', 10, 'number of training epochs')
cmd:option('-eta', 1, 'learning rate (eta)')
cmd:option('-batch_size', 32, 'batch size')

-- NN options
cmd:option('-embedding', 30, 'embedding size')
cmd:option('-gram', 5, 'gram size')
cmd:option('-hidden', 100, 'hidden layer size')
cmd:option('-direct', 0, '1 if direct connection layer')
cmd:option('-regularize', 1, '1 if regluarize embeddings')

params = cmd:parse(arg)

-- For renormalizing the lookup tables
function renorm(data, th)
    local size = data:size(1)
    for i = 1, size do
        local norm = data[i]:norm()
        if norm > th then
            data[i]:div(norm/th)
        end
    end
end

------------------------
-- Read in data
------------------------
myFile = hdf5.open(params['datafile'], 'r')
train = myFile:read('train'):all()
train_t = myFile:read('train_t'):all()
test = myFile:read('test'):all()
test_t = myFile:read('test_t'):all()
valid = myFile:read('valid'):all()
valid_t = myFile:read('valid_t'):all()

nV = 10000
c = params['gram']
d = params['embedding']
h = params['hidden']

batch_size = params['batch_size']
learning_rate = params['eta']

if params['cuda'] >= 1 then
    print("Using GPU")
    require("cutorch")
    require("cunn")
end

------------------------
-- Build the model
------------------------
local model = nn.Sequential()
local W_word = nn.LookupTable(nV, d)
local reshape = nn.View(batch_size, c * d)
model:add(W_word):add(reshape)
if params['direct'] > 0 then
    direct = nn.ConcatTable()
    W1 = nn.Linear(c * d, h)
    iden = nn.Identity()
    direct:add(W1):add(iden)
    W2 = nn.Linear(h + (c*d), nV)
    model:add(direct):add(nn.JoinTable(2)):add(nn.Tanh()):add(W2)
else
    W1 = nn.Linear(c * d, h)
    W2 = nn.Linear(h, nV)
    model:add(W1):add(nn.Tanh()):add(W2)
end
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

if params['cuda'] >= 1 then
    model:cuda()
    criterion:cuda()
end

function validate(inputs, outputs)
    local sum = 0
    local size = 0
    for i = 1,inputs:size(1)/batch_size do
        local input = inputs:narrow(1, (i-1)*batch_size+1, batch_size)
        local target = outputs:narrow(1, (i-1)*batch_size+1, batch_size)
        if params['cuda'] >= 1 then
            input = input:cuda()
            target = target:cuda()
        end
        local out = model:forward(input)
        local probs = out:exp()
        for j = 1, batch_size do
            local prob = probs[j][target[j]]
            sum = sum + math.log(prob)   
        end
        size = size + input:size(1)
    end
    return math.exp((-1/inputs:size(1)) * sum)
end

------------------------
-- Train the model
------------------------
model:reset()
last_perp = 0

for epoch = 1, params['nepochs'] do
    nll = 0
    for j = 1, train:size(1)/batch_size do
        model:zeroGradParameters()
        input = train:narrow(1, (j-1)*batch_size+1, batch_size)
        target = train_t:narrow(1, (j-1)*batch_size+1, batch_size)

        if params['cuda'] >= 1 then
            input = input:cuda()
            target = target:cuda()
        end
        
        out = model:forward(input)
        nll = nll + criterion:forward(out, target)

        deriv = criterion:backward(out, target)
        model:backward(input, deriv)
        model:updateParameters(learning_rate)
    end

    -- Calculate the perplexity, if it has increased since last
    -- epoch, half the learning rate
    perplexity = validate(valid, valid_t)
    if last_perp ~= 0 and perplexity > last_perp then
        learning_rate = learning_rate / 2
    end
    last_perp = perplexity

    -- Renormalize the weights of the lookup table
    if params['renorm'] == 1 then
        renorm(E.weight, 1) -- th = 1 taken from Sasha's code
    end
    print("Epoch:", epoch, "Loss:", nll, "Valid ppl:", perplexity)
end


----------------------
-- Compute predictions for test
----------------------

preds = torch.LongTensor(test:size(1))
for j = 1, test:size(1)/batch_size do
    input = test:narrow(1, (j-1)*batch_size+1, batch_size)
    if params['cuda'] >= 1 then input = input:cuda() end
    out = model:forward(input)
    y,i = torch.max(out:float(), 2)
    for k = 1, batch_size do
        preds[(j-1)*batch_size+1+k] = i[k]
    end
end

file = hdf5.open(params['predfile'],"w")
file:write("predictions", preds)
file:close()

torch.save(params['modelfile'], model)
