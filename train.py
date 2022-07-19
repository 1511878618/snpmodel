from .utils import *
import time 
from sklearn.metrics import accuracy_score
def test(net, dataIter, loss, device=try_gpu()):
    net.eval()
    
    test_accumulator = Accumulator(3)  #num, l, TP_TN
    
    iter_batch_num = len(dataIter)
    for _, x, y in dataIter:
        x, y = x.to(device), y.to(device)  # 送到device上
        
        output = net(x).flatten()
        
        test_accumulator.add(x.shape[0], loss(output, y.float()).item(), accuracy(y.cpu().detach().numpy(), output.cpu().detach().numpy(), normalize=False))
        
    test_l = test_accumulator[1]/iter_batch_num
    test_acc = test_accumulator[2]/test_accumulator[0]
    return test_l, test_acc


def train(net, train_dataIter, test_dataIter, optim, loss, summary_writer, epochs, savemodel=True, save_path="models", device=try_gpu()):

    net.to(device)
    iter_batch_num = len(train_dataIter)
    modelName = type(net).__name__


    if savemodel:
        model_root_path = os.path.join(save_path, modelName)
        mkdirs(model_root_path)
    for epoch in range(1, epochs + 1):
        net.train()
        t_start = time.time()
        accumulator = Accumulator(3)  # 句子个数, loss, TP+TN
        
        for _, x, y in train_dataIter:
            x, y = x.to(device), y.to(device)  # 送到device上
            optim.zero_grad()  # 梯度归零
            output = net(x).flatten()  # forward
            
            l = loss(output, y.float())  # 计算loss
            l.backward()  # 反向梯度计算
            optim.step()  # 梯度更新
            
            # accumulator.add(x.shape[0], l.cpu().detach().item())
            accumulator.add(x.shape[0], l.item(), accuracy(y.cpu().detach().numpy(), output.cpu().detach().numpy(), normalize=False))
            
        t_end = time.time()
        
        #  计算acc loss
        train_loss_epoch = accumulator[1]/iter_batch_num
        train_acc_epoch = accumulator[2]/accumulator[0]
        test_loss, test_acc = test(net, test_dataIter, loss, device)
        #  tensorboard记录 acc loss
        summary_writer.add_scalar(f"{modelName}_train_loss", train_loss_epoch, epoch)
        summary_writer.add_scalar(f"{modelName}_train_acc", train_acc_epoch, epoch)
        summary_writer.add_scalar(f"{modelName}_test_loss", test_loss, epoch)
        summary_writer.add_scalar(f"{modelName}_test_acc", test_acc, epoch)
        
        # save
        if savemodel:
            epoch_model_path = os.path.join(model_root_path, f"{modelName}_{epoch}.pt")
            torch.save(net, epoch_model_path)
        
        #  显示部分输出
        if epoch % 10 == 0:
            print("epoch {} : train: mean loss/per {:.3f} acc is {:.3f} and time is {}review/per sec".format(epoch, train_loss_epoch, train_acc_epoch, accumulator[0]/(t_end - t_start)))
            print(f"test: loss {test_loss:.3f} and acc is {test_acc:.3f}")
 

        
def predict(dataIter, model, device=try_gpu()):
    out = []
    for x in dataIter:
        x = x.to(device)
        o = model(x).cpu().detach()
        out.append(o)
        
    return torch.concat(out)