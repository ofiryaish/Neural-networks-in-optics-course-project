import common
import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if (scale == 3):
            if (args.scale_type == 'upConv'):
                m_tail = [
                    nn.ConvTranspose2d(n_feats, n_feats, 5, stride=3, padding=1), # H_output = (H_input-1)*strides_size - 2*padding_size + kernel_size when assuming other parameters are default. The same for W_output
                    #nn.Upsample(scale_factor=scale, mode='nearest'),
                    conv(n_feats, args.o_colors, kernel_size)
                        ]
            elif (args.scale_type == 'upNeighbor'):
                m_tail = [
                    #nn.ConvTranspose2d(n_feats, n_feats, 5, stride=3, padding=1), # H_output = (H_input-1)*strides_size - 2*padding_size + kernel_size when assuming other parameters are default. The same for W_output
                    nn.Upsample(scale_factor=scale, mode='nearest'),
                    conv(n_feats, args.o_colors, kernel_size)
                        ]
        else: 
            #assume power of two
            m_tail = [
                common.Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.o_colors, kernel_size)
                     ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x
