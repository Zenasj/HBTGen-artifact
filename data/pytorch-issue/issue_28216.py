import torch
import torch.nn as nn

model = models.resnet18(pretrained=False)
model.train()
optimizer = Adam(model.parameters(), lr=1E-2)
lr_scheduler = OneCycleLR(optimizer, 
                          max_lr=1E-2, 
                          total_steps=100, 
                          cycle_momentum=False,
                          div_factor=25)

lr = []
for epoch in range(100):
    optimizer.zero_grad()
    optimizer.step()
    lr_scheduler.step()
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
plt.plot(lr);

lr_scheduler = OneCycleLR(optimizer, 
                          max_lr=1E-2, 
                          total_steps=100, 
                          cycle_momentum=False,
                          div_factor=10000)

def test_onecycle_lr_div_factors(self):
        # https://github.com/pytorch/pytorch/issues/28216

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        div_factors = [1, 10, 25, 100]
        final_div_factors = [1, 1E2, 1E4, 1E6]

        for div_factor, final_div_factor in zip(div_factors, final_div_factors):
            for strategy in ['linear', 'cos']:
                scheduler = OneCycleLR(optimizer, max_lr=25, total_steps=10,
                                       anneal_strategy=strategy,
                                       cycle_momentum=False,
                                       div_factor=div_factor,
                                       final_div_factor=final_div_factor
                                       )
                lrs = []
                for i in range(10):
                    lrs.append(optimizer.param_groups[0]["lr"])
                    optimizer.step()
                    scheduler.step()

                initial_lr, max_lr, last_lr = lrs[0], max(lrs), lrs[-1]
                self.assertAlmostEqual(max_lr / initial_lr, div_factor)
                self.assertAlmostEqual(initial_lr, final_div_factor * last_lr)

def test_onecycle_lr_div_factors(self):
        # https://github.com/pytorch/pytorch/issues/28216

        model = torch.nn.Linear(2, 1)
        div_factors = [1, 10, 25, 100]
        final_div_factors = [1, 1E2, 1E4, 1E6]

        for div_factor, final_div_factor in zip(div_factors, final_div_factors):
            for strategy in ['linear', 'cos']:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
                scheduler = OneCycleLR(optimizer, max_lr=25, total_steps=10,
                                       anneal_strategy=strategy,
                                       cycle_momentum=False,
                                       div_factor=div_factor,
                                       final_div_factor=final_div_factor
                                       )
                lrs = []
                for i in range(10):
                    lrs.append(optimizer.param_groups[0]["lr"])
                    optimizer.step()
                    scheduler.step()

                initial_lr, max_lr, last_lr = lrs[0], max(lrs), lrs[-1]
                self.assertAlmostEqual(max_lr / initial_lr, div_factor)
                self.assertAlmostEqual(initial_lr, final_div_factor * last_lr)