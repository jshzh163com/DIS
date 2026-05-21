import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.utils.weight_norm as weightNorm


class CNN(nn.Module):
    def __init__(self, pretrained=False):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))

        self.layer5 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x


class CNN_fea(nn.Module):
    def __init__(self, pretrained=False):
        super(CNN_fea, self).__init__()
        self.model_cnn = CNN()
        self.in_features = self.model_cnn.layer5[0].out_features

    def forward(self, x):
        return self.model_cnn(x)


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        return self.fc(x)


def domain_proto_nce_loss(features, domain_labels, temperature=0.1):
    device = features.device
    features = F.normalize(features, dim=1)

    unique_domains, inverse = torch.unique(domain_labels, return_inverse=True)
    K = unique_domains.size(0)
    C = features.size(1)

    protos = torch.zeros(K, C, device=device)
    counts = torch.zeros(K, device=device)

    protos.index_add_(0, inverse, features)
    counts.index_add_(0, inverse, torch.ones_like(
        inverse, dtype=features.dtype))

    protos = protos / counts.unsqueeze(1).clamp_min(1.0)
    protos = F.normalize(protos, dim=1)

    logits = torch.matmul(features, protos.t()) / temperature
    loss = F.cross_entropy(logits, inverse)
    return loss


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        super(Discriminator, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DIS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.featurizer = CNN_fea(args)

        self.tfbd = args.bottleneck // 2
        self.bottleneck = feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)

        self.classifier = feat_classifier(
            args.num_classes, self.tfbd, args.classifier)
        self.dclassifier = feat_classifier(
            (args.domain_num - len(args.test_envs)), self.tfbd, args.classifier)
        self.discriminator = Discriminator(
            self.tfbd, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.cdiscriminator = Discriminator(
            self.tfbd, args.dis_hidden, args.num_classes)

        self.teaf = CNN_fea(args)
        self.teab = feat_bottleneck(
            self.teaf.in_features, self.tfbd, args.layer)
        self.teac = feat_classifier(
            args.domain_num - len(args.test_envs), self.tfbd, args.classifier)
        self.teaNet = nn.Sequential(
            self.teaf,
            self.teab,
            self.teac
        )

    def teanettrain(self, dataloaders, epochs, opt1, sch1, steps_per_epoch):
        self.teaNet.train()
        device = self.args.device

        train_iters = [iter(loader) for loader in dataloaders]

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                minibatches = [next(train_iter) for train_iter in train_iters]

                all_x = torch.cat(
                    [data[0].to(device).float() for data in minibatches], dim=0
                )
                all_dy = torch.cat([
                    torch.full((data[0].shape[0],), i,
                               dtype=torch.int64, device=device)
                    for i, data in enumerate(minibatches)
                ], dim=0)

                perm = torch.randperm(all_x.size(0), device=all_x.device)
                all_x = all_x[perm]
                all_dy = all_dy[perm]

                feat = self.teab(self.teaf(all_x))
                logits = self.teac(feat)

                loss_dom = F.cross_entropy(logits, all_dy)
                loss_proto = domain_proto_nce_loss(
                    feat, all_dy, temperature=0.1)
                loss = 0.5 * loss_dom + loss_proto

                opt1.zero_grad()
                loss.backward()
                opt1.step()

            if sch1 is not None:
                sch1.step()

        self.teaNet.eval()

    def update(self, minibatches, opt, sch, epoch):
        device = self.args.device
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])

        all_dy = torch.cat([
            torch.full((data[0].shape[0],), i,
                       dtype=torch.int64, device=device)
            for i, data in enumerate(minibatches)
        ])

        with torch.no_grad():
            tfea = self.teab(self.teaf(all_x)).detach()

        shared_fea = self.featurizer(all_x)
        z = self.bottleneck(shared_fea)
        all_ifea = z[:, :self.tfbd]
        all_cfea = z[:, self.tfbd:]

        cls_logits = self.classifier(all_ifea)
        loss_cls = F.cross_entropy(cls_logits, all_y)

        loss_mse = F.mse_loss(all_cfea, tfea) * self.args.alpha

        dcls_logits = self.dclassifier(all_cfea)
        loss_dcls = F.cross_entropy(dcls_logits, all_dy) * self.args.beta

        grl_lambda_dom = getattr(self.args, "grl_lambda_dom", 1.0)
        grl_lambda_cls = getattr(self.args, "grl_lambda_cls", 1.0)
        lambda_d_adv = getattr(self.args, "lambda_d_adv", 0.1)
        lambda_cls_adv = getattr(self.args, "lambda_cls_adv", 0.1)

        inv_rev = ReverseLayerF.apply(all_ifea, grl_lambda_dom)
        dom_logits_adv = self.discriminator(inv_rev)
        loss_d_adv = F.cross_entropy(dom_logits_adv, all_dy) * lambda_d_adv

        spec_rev = ReverseLayerF.apply(all_cfea, grl_lambda_cls)
        cls_logits_adv = self.cdiscriminator(spec_rev)
        loss_cls_adv = F.cross_entropy(
            cls_logits_adv, all_y) * lambda_cls_adv

        cos = F.cosine_similarity(all_ifea, all_cfea, dim=1)
        loss_cos = torch.mean(cos ** 2) * self.args.lam

        loss = (
            loss_cls
            + loss_mse
            + loss_dcls
            + loss_d_adv
            + loss_cls_adv
            + loss_cos
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()

        return {
            'cls': loss_cls.item(),
            'kd': loss_mse.item(),
            'd_cls': loss_dcls.item(),
            'd_adv': loss_d_adv.item(),
            'cls_adv': loss_cls_adv.item(),
            'dissimilarity': loss_cos.item(),
            'total': loss.item()
        }

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x))[:, :self.tfbd])
