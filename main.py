import logging
import os
import time
import torch


from category_id_map import CATEGORY_ID_LIST
from tricks.EMA import EMA
from tricks.FGM import FGM
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal, MultiModal_a, MultiModal_b
from util import setup_device, setup_seed, setup_logging, build_optimizer_raw, build_optimizer_boost, evaluate

@staticmethod
def cal_loss(prediction, label):
    label = label.squeeze(dim=1)
    loss = F.cross_entropy(prediction, label)
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]
    return loss, accuracy, pred_label_id, label

def validate(model, val_dataloader):
    
    model.eval()
    predictions_a = []
    labels_a = []
    predictions_b = []
    labels_b = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch={key:batch[key].cuda() for key in batch}
            prediction_a = model.module.model_a(batch)
            prediction_b = model.module.model_b(batch)

            _, _, pred_label_id_a, label_a = model.module.cal_loss(prediction_a, batch['label'])
            _, _, pred_label_id_b, label_b = model.module.cal_loss(prediction_b, batch['label'])

            predictions_a.extend(pred_label_id_a.cpu().numpy())
            labels_a.extend(label_a.cpu().numpy())
            predictions_b.extend(pred_label_id_b.cpu().numpy())
            labels_b.extend(label_b.cpu().numpy())

    results_a = evaluate(predictions_a, labels_a)
    results_b = evaluate(predictions_b, labels_b)

    model.train()
    return results_a, results_b


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader, unlabel_dataloader = create_dataloaders(args)
    iter_unlabel_dataloader = iter(unlabel_dataloader)

    # 2. build model and optimizers
    model_a = MultiModal_a(args, len(CATEGORY_ID_LIST))
    model_b = MultiModal_b(args, len(CATEGORY_ID_LIST))

    restore_checkpoint(model_a, args.raw_ckpt)
    restore_checkpoint(model_b, args.raw_ckpt)

    model = MultiModal(model_a, model_b)
    optimizer_a, scheduler_a = build_optimizer_raw(args, model_a)
    optimizer_b, scheduler_b = build_optimizer_boost(args, model_b)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score_a = args.best_score
    best_score_b = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    ema = EMA(model, 0.999)
    ema.register()
    # ema_b = EMA(model_b, 0.999)
    # ema_b.register()

    fgm = FGM(model)

    for epoch in range(args.max_epochs):

        for batch in train_dataloader:
            try:
                unlabel_inputs = next(iter_unlabel_dataloader, None)
                
                model.train()
                loss_a, un_loss_a, loss_b, un_loss_b, accuracy_a, accuracy_b = model(batch, unlabel_inputs)

                loss_a = loss_a.mean()
                un_loss_a = un_loss_a.mean()
                loss_b = loss_b.mean()
                un_loss_b = un_loss_b.mean()

                accuracy_a = accuracy_a.mean()
                accuracy_b = accuracy_b.mean()

                a_loss = loss_a + un_loss_a
                b_loss = loss_b + un_loss_b

                a_loss.backward() 
                b_loss.backward()

                fgm.attack()

                adv_loss_a, adv_un_loss_a, adv_loss_b, adv_un_loss_b, adv_accuracy_a, adv_accuracy_b = model(batch, unlabel_inputs)

                adv_loss_a = adv_loss_a.mean()
                adv_un_loss_a = adv_un_loss_a.mean()
                adv_loss_b = adv_loss_b.mean()
                adv_un_loss_b = adv_un_loss_b.mean()

                adv_a_loss = adv_loss_a + adv_un_loss_a
                adv_b_loss = adv_loss_b + adv_un_loss_b

                adv_a_loss.backward()
                adv_b_loss.backward()

                fgm.restore()

                optimizer_a.step()
                optimizer_b.step()

                ema.update()

                optimizer_a.zero_grad()
                scheduler_a.step()
                optimizer_b.zero_grad()
                scheduler_b.step()


                step += 1
                if step % args.print_steps == 0:
                    
                    # ema_b.apply_shadow()
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss_a {loss_a:.3f}, un_loss_a {un_loss_a:.3f}, loss_b {loss_b:.3f}, un_loss_b {un_loss_b:.3f}, accuracy_a {accuracy_a:.3f}, accuracy_b {accuracy_b:.3f}")

                # 4. validation
                if step % 1000 == 0:
                    print('Begin Validation')
                    ema.apply_shadow()
                    # print(model.module.model_a)
                    results_a, results_b= validate(model, val_dataloader)
                    results_a = {k: round(v, 4) for k, v in results_a.items()}
                    results_b = {k: round(v, 4) for k, v in results_b.items()}
                    logging.info(f"Epoch {epoch} step {step}:, {results_a}, {results_b}")

                    # 5. save checkpoint
                    mean_f1_a = results_a['mean_f1']
                    if mean_f1_a > best_score_a or mean_f1_a > 0.99:
                        best_score_a = mean_f1_a
                        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1_a},
                                f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1_a}.bin')
                    
                    mean_f1_b = results_b['mean_f1']
                    if mean_f1_b > best_score_b or mean_f1_b > 0.99:
                        best_score_b = mean_f1_b
                        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1_b},
                                f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1_b}.bin')
                    ema.restore()
            
            except:
                print('The length is not enough!')
                _, _, unlabel_dataloader = create_dataloaders(args)
                iter_unlabel_dataloader = iter(unlabel_dataloader)
                print('We fix it!')
                # ema_a.restore()
                # ema_b.restore()

        
        
        # exit()
def restore_checkpoint(model_to_load, restore_name='BEST_EVAL_LOSS'):
    # restore_bin = str(f'{args.savedmodel_path}/{restore_name}.pth')
    state_dict = torch.load(restore_name)['model']

    own_state = model_to_load.state_dict()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if name not in own_state:
            logging.info(f'Skipped: {name}')
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            logging.info(f"Successfully loaded: {name}")
        except:
            pass
            logging.info(f"Part load failed: {name}")

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
