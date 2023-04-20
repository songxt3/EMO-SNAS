"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import data_loader
import os
from datetime import datetime
from utils import StatusUpdateTool
import multiprocessing

TOTAL_SPIKE_RATE = 0
NUM_OF_SPIKES = 0
AVG_OF_SPIKES = 0

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class LIFNode(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFNode, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_LIF(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_LIF, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.spike = LIFNode()

    def forward(self, x):
        spike_num = 0
        identity = x

        out = self.conv1_s(x)
        out = self.spike(out)
        global TOTAL_SPIKE_RATE, NUM_OF_SPIKES
        spike_num += out.sum().item()
        # spike_rate_lif1 = float(torch.count_nonzero(out).item() / out.numel())

        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.spike(out)
        spike_num += out.sum().item()
        spike_num /= x.shape[0]
        # spike_rate_lif2 = float(torch.count_nonzero(out).item() / out.numel())

        # todo:calculate spike rate
        # TOTAL_SPIKE_RATE = TOTAL_SPIKE_RATE + spike_rate_lif1 + spike_rate_lif2
        NUM_OF_SPIKES = NUM_OF_SPIKES + spike_num
        # print('current spikes number:%d, number of spikes:%d'%(spike_num, NUM_OF_SPIKES))

        return out


class BasicBlock_LIF_backward(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        self.last_in = 0

        super(BasicBlock_LIF_backward, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.spike = LIFNode()

    def forward(self, x):
        spike_num = 0
        identity = x
        out = self.conv1_s(x)
        out = self.spike(out)
        dim1 = out[:, 0:1, :, :]
        dim2 = out[:, 1:2, :, :]
        dim3 = dim1.mul(dim2)
        ans = torch.cat((out, dim3), 1)
        out = ans
        global TOTAL_SPIKE_RATE, NUM_OF_SPIKES
        spike_num += out.sum().item()
        # spike_rate_lif1 = float(torch.count_nonzero(out).item() / out.numel())

        out = self.conv2_s(out)
        dim1_ = out[:, 0:1, :, :]
        dim2_ = out[:, 1:2, :, :]
        dim3_ = out[:, 2:3, :, :]
        dim1_ = dim1_.mul(dim3_)
        dim2_ = dim2_.mul(dim3_)
        ans_ = torch.cat((dim1_, dim2_), 1)
        out = ans_

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.spike(out)
        spike_num += out.sum().item()
        spike_num /= x.shape[0]
        # spike_rate_lif2 = float(torch.count_nonzero(out).item() / out.numel())

        self.last_in = out

        # todo:calculate spike rate
        # TOTAL_SPIKE_RATE = TOTAL_SPIKE_RATE + spike_rate_lif1 + spike_rate_lif2
        NUM_OF_SPIKES = NUM_OF_SPIKES + spike_num
        # print('current spikes number:%d, number of spikes:%d'%(spike_num, NUM_OF_SPIKES))

        return out

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


class EvoSNNModel(nn.Module):
    def __init__(self, groups=1, width_per_group=64, norm_layer=None):
        super(EvoSNNModel, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 3
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.T = 2
        #generated_init

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()

    def _forward_impl(self, x):
        #generate_forward

        out = self.finalpool(out)
        out = torch.flatten(out, 2)
        out = self.linear(out)

        return out

    def forward(self, x):
        x = add_dimention(x, self.T)
        return self._forward_impl(x)


class TrainModel(object):
    def __init__(self):
        data_dir = os.path.expanduser('dataset/tiny-imagenet-200/')
        train_loader, test_loader = data_loader.get_train_valid_loader(data_dir, batch_size=64, valid_size=0.1, shuffle=True, random_seed=2312390, show_sample=False, num_workers=1, pin_memory=True, dataset='tinyimagenet')
        model = EvoSNNModel()
        total_param = sum([param.nelement() for param in model.parameters()])
        cudnn.benchmark = True
        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        max_test_accuracy = 0
        train_epoch = 0

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.train_epoch = train_epoch
        self.max_test_accuracy = max_test_accuracy
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.total_param = total_param

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch):
        print('Fileid:%s, Train:' % self.file_id)
        lr = 0.001
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        epochs = StatusUpdateTool.get_epoch_size()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)

        self.model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(self.train_loader):
            optimizer.zero_grad()
            labels = labels.cuda()
            images = images.cuda()
            outputs = self.model(images)
            mean_out = outputs.mean(1)
            loss = self.criterion(mean_out, labels)
            running_loss += loss.item()
            loss.mean().backward()
            optimizer.step()
            total += float(labels.size(0))
            _, predicted = mean_out.cpu().max(1)
            correct += float(predicted.eq(labels.cpu()).sum().item())

            global NUM_OF_SPIKES, AVG_OF_SPIKES
            if AVG_OF_SPIKES == 0:
                AVG_OF_SPIKES = NUM_OF_SPIKES
            else:
                AVG_OF_SPIKES = (AVG_OF_SPIKES + NUM_OF_SPIKES) / 2
            # print('Number of spikes:%d, average of spikes:%d' % (NUM_OF_SPIKES, AVG_OF_SPIKES))
            NUM_OF_SPIKES = 0
        print('Fileid:%s,  Train-Epoch:%3d,  Loss: %.3f,  Acc:%.3f' % (self.file_id, epoch + 1, running_loss, (correct / total)))
        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f' % (epoch + 1, running_loss, (correct / total)))
        self.train_epoch += 1
        scheduler.step()
        return running_loss, 100 * correct / total

    def test(self):
        with torch.no_grad():
            print('Fileid:%s, Test:' % self.file_id)
            self.model.eval()
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                mean_out = outputs.mean(1)
                _, predicted = mean_out.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(self.test_loader), ' Acc: %.5f' % acc)
            final_acc = 100 * correct / total

            if self.max_test_accuracy < final_acc:
                self.max_test_accuracy = final_acc
                print('Fileid:%s, Max-Acc:%.3f' % (self.file_id, self.max_test_accuracy))
                self.log_record('Max-Acc:%.3f' % self.max_test_accuracy)
                # torch.save(self.model.state_dict(), './logs/model_bestT1_cifar10_r11.pth.tar')
                # print('保存模型参数', './logs/' + log_prefix + '/model_bestT1_cifar10_r11.pth.tar')
            print('Fileid:%s,  Validate-Acc:%.3f, epoch:%d' % (self.file_id, final_acc, self.train_epoch))
            self.log_record('Validate-Acc:%.3f, epoch:%d' % (final_acc, self.train_epoch))

            return final_acc

    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        print("Number of parameter: %.2fM" % (self.total_param / 1e6))
        self.log_record("Number of parameter: %.2fM" % (self.total_param / 1e6))
        for p in range(total_epoch):
            self.train(p)
            global TOTAL_SPIKE_RATE, NUM_OF_SPIKES, AVG_OF_SPIKES
            # if AVG_OF_SPIKES == 0:
            #     AVG_OF_SPIKES = NUM_OF_SPIKES
            # else:
            #     AVG_OF_SPIKES = (AVG_OF_SPIKES + NUM_OF_SPIKES) / 2
            print('Average of spikes:%d' % (AVG_OF_SPIKES))
            self.log_record('Average of spikes:%d' % (AVG_OF_SPIKES))
            NUM_OF_SPIKES = 0
            self.test()
        return self.max_test_accuracy


class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        m = TrainModel()
        try:
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            best_acc = m.process()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            # TODO: random weight GA
            print('Finished-Acc:%.3f,Average number of spikes:%d'%(best_acc, AVG_OF_SPIKES))
            m.log_record('Finished-Acc:%.3f,Average number of spikes:%d'%(best_acc, AVG_OF_SPIKES))
            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s=%.5f,%.5f,%.5f\n'%(file_id, (100-best_acc), AVG_OF_SPIKES, -1.0))
            f.flush()
            f.close()
"""