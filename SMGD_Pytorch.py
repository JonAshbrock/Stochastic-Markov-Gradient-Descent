import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import time

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=240,shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(28*28*1, 300)
        self.fc2=nn.Linear(300,150)
        self.fc3=nn.Linear(150,10)
        # Now we must alter the weights to be initialized in a quantized state
        for f in self.parameters():
            f.data= torch.clamp(delta*(-0.5+torch.round(f.data/delta +0.5)), min=-cutoff,max=cutoff)

        
    def forward(self, x):
        x=x.view(-1,self.num_flat_features(x))
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features =1
        for s in size:
            num_features *= s
        return num_features
    
    def test(self):
        self.correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = my_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                self.correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * self.correct / total))


bits = 1
delta=0.1
normalizer = 10# higher normalizer --> fewer updates

#derived constantsmy_net.
cutoff = (2**(bits-1)-0.5)*delta

my_net=Net()
criterion  =nn.CrossEntropyLoss()
optimizer=optim.SGD(my_net.parameters(), lr=0.001)


for epoch in range(120):  # loop over the dataset multiple times
    #normalizer=epoch+1
    running_loss = 0.0
    t0=time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = my_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for f in my_net.parameters():
            m=torch.distributions.Bernoulli(torch.clamp(input=torch.abs(f.grad.data)/normalizer, min=0,max=1))
            f.data.sub_(m.sample()*torch.sign(f.grad.data)*delta)
            torch.clamp_(f.data, min=-cutoff,max=cutoff)

        # print statistics
        running_loss += loss.item()
        if i % 250 == 249:    # print every 2000 mini-batches
            print(time.time()-t0)
            t0=time.time()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 250))
            running_loss = 0.0

print('Finished Training')


#my_net.test()
