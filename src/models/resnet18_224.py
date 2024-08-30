import torch
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18_224(ResNet):
    def __init__(self,
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],
                 num_classes=1000):
        super(ResNet18_224, self).__init__(block=block,
                                               layers=layers,
                                               num_classes=num_classes)
        self.feature_size = 512

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)

        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc
    


# all the available checkpoints based on dataset
datasets_checkpoints = {
    'ImageNet-200-s0' : '../../models/ImageNet-200/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt',
    'ImageNet-200-s1' : '../../models/ImageNet-200/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1/best.ckpt',
    'ImageNet-200-s2' : '../../models/ImageNet-200/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best.ckpt',

}

def load_pretrained_weights_224(dataset='ImageNet-200', model_version='s0', num_classes=200):
    # get the full checkpoint
    full_checkpoint = dataset + '-' + model_version
    checkpoint_path = datasets_checkpoints[full_checkpoint]
    model = ResNet18_224(num_classes=num_classes)
    # Charger les poids pré-entraînés
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Charger le dictionnaire d'état dans le modèle
    model.load_state_dict(state_dict)
    # print the model architecture
    print("Model pretrained weight have been successfully loaded !")
    return model


