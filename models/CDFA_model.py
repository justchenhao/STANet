import torch
import itertools
from .base_model import BaseModel
from . import backbone
import torch.nn.functional as F
from . import loss


class CDFAModel(BaseModel):
    """
    change detection module:
    feature extractor+ spatial-temporal-self-attention
    contrastive loss
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['f']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if opt.phase == 'test':
            self.istest = True
        self.visual_names = ['A', 'B', 'L', 'pred_L_show']  # visualizations for A and B
        if self.istest:
            self.visual_names = ['A', 'B', 'pred_L_show']  # visualizations for A and B

        self.visual_features = ['feat_A','feat_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['F','A']
        else:  # during test time, only load Gs
            self.model_names = ['F','A']
        self.istest = False
        self.ds = 1
        self.n_class =2
        self.netF = backbone.define_F(in_c=3, f_c=opt.f_c, type=opt.arch).to(self.device)
        self.netA = backbone.CDSA(in_c=opt.f_c, ds=opt.ds, mode=opt.SA_mode).to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionF = loss.BCL()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                self.netF.parameters(),
            ), lr=opt.lr*opt.lr_decay, betas=(opt.beta1, 0.999))
            self.optimizer_A = torch.optim.Adam(self.netA.parameters(), lr=opt.lr*1, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_A)


    def set_input(self, input):
        self.A = input['A'].to(self.device)
        self.B = input['B'].to(self.device)
        if self.istest is False:
            if 'L' in input.keys():
                self.L = input['L'].to(self.device).long()

        self.image_paths = input['A_paths']
        if self.isTrain:
            self.L_s = self.L.float()
            self.L_s = F.interpolate(self.L_s, size=torch.Size([self.A.shape[2]//self.ds, self.A.shape[3]//self.ds]),mode='nearest')
            self.L_s[self.L_s == 1] = -1  # change
            self.L_s[self.L_s == 0] = 1  # no change


    def test(self, val=False):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            if val:  # 返回score
                from util.metrics import RunningMetrics
                metrics = RunningMetrics(self.n_class)
                pred = self.pred_L.long()

                metrics.update(self.L.detach().cpu().numpy(), pred.detach().cpu().numpy())
                scores = metrics.get_cm()
                return scores
            else:
                return self.pred_L.long()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.feat_A = self.netF(self.A)  # f(A)
        self.feat_B = self.netF(self.B)   # f(B)

        self.feat_A, self.feat_B = self.netA(self.feat_A,self.feat_B)

        self.dist = F.pairwise_distance(self.feat_A, self.feat_B, keepdim=True)  # 特征距离

        self.dist = F.interpolate(self.dist, size=self.A.shape[2:], mode='bilinear',align_corners=True)

        self.pred_L = (self.dist > 1).float()
        # self.pred_L = F.interpolate(self.pred_L, size=self.A.shape[2:], mode='nearest')
        self.pred_L_show = self.pred_L.long()

        return self.pred_L

    def backward(self):
        self.loss_f = self.criterionF(self.dist, self.L_s)

        self.loss = self.loss_f
        # print(self.loss)
        self.loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute feat and dist

        self.set_requires_grad([self.netF, self.netA], True)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_A.zero_grad()
        self.backward()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.optimizer_A.step()