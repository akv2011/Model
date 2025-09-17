"""AQCF-Net Training Script"""

import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from model_qcfnet import AQCF_Net, create_aqcfnet
from data_loading import setup_data_loaders, compute_dice_coefficient_per_class
from config import OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, INPUT_CHANNELS, OUTPUT_CLASSES, CHANNELS, STRIDES


def validation_qcf_net(model, val_loader_ct, val_loader_mri, device, post_pred, post_label, dice_metric):
    model.eval()
    with torch.no_grad():
        for batch in val_loader_ct:
            val_inputs_ct = batch["image"].to(device)
            val_labels_ct = batch["label"].to(device)
            
            predictor_ct = lambda data: model(x_in=data, modality="CT")
            pred_ct = sliding_window_inference(
                val_inputs_ct, 
                (256, 256, 16), 
                4, 
                predictor_ct, 
                overlap=0.8
            )
            
            labels_list = decollate_batch(val_labels_ct)
            labels_convert = [post_label(l) for l in labels_list]
            outputs_list = decollate_batch(pred_ct)
            output_convert = [post_pred(o) for o in outputs_list]
            
            dice_metric(y_pred=output_convert, y=labels_convert)
        
        mean_dice_ct = dice_metric.aggregate().item()
        dice_metric.reset()
        for batch in val_loader_mri:
            val_inputs_mri = batch["image"].to(device)
            val_labels_mri = batch["label"].to(device)
            
            predictor_mri = lambda data: model(x_in=data, modality="MRI")
            pred_mri = sliding_window_inference(
                val_inputs_mri, 
                (256, 256, 16), 
                4, 
                predictor_mri, 
                overlap=0.8
            )
            
            labels_list = decollate_batch(val_labels_mri)
            labels_convert = [post_label(l) for l in labels_list]
            outputs_list = decollate_batch(pred_mri)
            output_convert = [post_pred(o) for o in outputs_list]
            
            dice_metric(y_pred=output_convert, y=labels_convert)
        
        mean_dice_mri = dice_metric.aggregate().item()
        dice_metric.reset()
        
    return mean_dice_ct, mean_dice_mri


def train_qcf_net(
    num_epochs, model, train_loader, val_loader_ct, val_loader_mri, 
    optimizer, loss_function, scheduler, device, post_pred, post_label, dice_metric,
    epoch_loss_values, metric_values_ct, metric_values_mri, 
    output_dir, model_name="best_metric_model.pth"
):
    
    best_metric = 0.0
    best_metric_epoch = -1
    model.to(device)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        model.train()
        epoch_loss = 0
        step = 0
        
        epoch_iterator = tqdm(
            train_loader, 
            desc=f"Training QCF-Net (Epoch {epoch+1})", 
            dynamic_ncols=True
        )
        
        for batch in epoch_iterator:
            step += 1
            
            ct_images = batch["ct_image"].to(device)
            ct_labels = batch["ct_label"].to(device)
            mri_images = batch["mri_image"].to(device)
            mri_labels = batch["mri_label"].to(device)

            optimizer.zero_grad()

            pred_ct, pred_mri = model(ct_images=ct_images, mri_images=mri_images)

            loss_ct = loss_function(pred_ct, ct_labels)
            loss_mri = loss_function(pred_mri, mri_labels)
            total_loss = loss_ct + loss_mri

            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_iterator.set_postfix({"train_loss": f"{total_loss.item():.4f}"})
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            print("Running validation...")
            dice_val_ct, dice_val_mri = validation_qcf_net(
                model, val_loader_ct, val_loader_mri, device, 
                post_pred, post_label, dice_metric
            )
            
            metric_values_ct.append(dice_val_ct)
            metric_values_mri.append(dice_val_mri)
            avg_dice_val = (dice_val_ct + dice_val_mri) / 2
            
            print(f"Validation Dice -> CT: {dice_val_ct:.4f}, MRI: {dice_val_mri:.4f}, Average: {avg_dice_val:.4f}")
            
            scheduler.step(avg_dice_val)
            
            if avg_dice_val > best_metric:
                best_metric = avg_dice_val
                best_metric_epoch = epoch + 1
                
                model_path = os.path.join(output_dir, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"Saved new best model! Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
                print(f"Model saved to: {model_path}")
        
    print(f"\\nTraining finished. Best average Dice achieved: {best_metric:.4f} at epoch {best_metric_epoch}")
    return epoch, best_metric, best_metric_epoch


def setup_training_components(device, output_classes, finetune=False, finetune_lr=1e-5):
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    

    post_label = AsDiscrete(to_onehot=output_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=output_classes)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    cudnn.benchmark = True
    
    print(f"\\n{'FINE-TUNING' if finetune else 'TRAINING'} SETUP COMPLETE")
    print(f"Loss function: DiceCELoss")
    print(f"Evaluation metric: DiceMetric")
    print(f"CuDNN benchmark: Enabled")
    
    return {
        'loss_function': loss_function,
        'post_pred': post_pred,
        'post_label': post_label,
        'dice_metric': dice_metric
    }


def create_model_and_optimizer(device, input_channels=1, output_classes=3, 
                              channels=(12, 24, 48, 96, 192), strides=(2, 2, 2, 2),
                              learning_rate=1e-4, weight_decay=1e-5):
   
    model = create_aqcfnet(
        input_channels=input_channels,
        output_classes=output_classes,
        quat_channels=channels,
        strides=strides
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=4, 
        verbose=True
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"Scheduler: ReduceLROnPlateau")
    
    return model, optimizer, scheduler


def run_training_pipeline(data_dir=".", dataset_json="dataset_0.json", 
                         num_epochs=500, batch_size=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("Setting up data loaders...")
    data_result = setup_data_loaders(data_dir, dataset_json, batch_size=batch_size)
    if data_result is None:
        print("Data loading failed. Please check your dataset configuration.")
        return
    
    train_loader = data_result['train_loader']
    val_loader_ct = data_result['val_loader_ct'] 
    val_loader_mri = data_result['val_loader_mri']
    
    print("\\nCreating model and optimizer...")
    model, optimizer, scheduler = create_model_and_optimizer(
        device=device,
        input_channels=INPUT_CHANNELS,
        output_classes=OUTPUT_CLASSES,
        channels=CHANNELS,
        strides=STRIDES
    )

    print("\\nSetting up training components...")
    components = setup_training_components(device, OUTPUT_CLASSES)
    

    epoch_loss_values = []
    metric_values_ct = []
    metric_values_mri = []
    
    
    try:
        last_epoch, best_metric, best_metric_epoch = train_qcf_net(
            num_epochs=num_epochs,
            model=model,
            train_loader=train_loader,
            val_loader_ct=val_loader_ct,
            val_loader_mri=val_loader_mri,
            optimizer=optimizer,
            loss_function=components['loss_function'],
            scheduler=scheduler,
            device=device,
            post_pred=components['post_pred'],
            post_label=components['post_label'],
            dice_metric=components['dice_metric'],
            epoch_loss_values=epoch_loss_values,
            metric_values_ct=metric_values_ct,
            metric_values_mri=metric_values_mri,
            output_dir=output_dir
        )
        
        print("\\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Best Dice Score: {best_metric:.4f} at epoch {best_metric_epoch}")
        print(f"Model saved in: {output_dir}")
        
        return {
            'model': model,
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'epoch_loss_values': epoch_loss_values,
            'metric_values_ct': metric_values_ct,
            'metric_values_mri': metric_values_mri
        }
        
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
        return None
    except Exception as e:
        print(f"\\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = run_training_pipeline(
        data_dir=".",
        dataset_json="dataset_0.json",
        num_epochs=50,
        batch_size=1
    )
    
    if result:
        print("Training pipeline completed successfully!")
    else:
        print("Training pipeline failed or was interrupted.")