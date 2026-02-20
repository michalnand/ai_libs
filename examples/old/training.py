import image_libs
import numpy
import torch


if __name__ == "__main__":

    #dataset_path = "/Users/michal/datasets/textures/"
    dataset_path = "/home/michal/datasets/textures/"

    images_loader = image_libs.ImagesLoader(dataset_path)
    images_aug    = image_libs.ImagesAug(images_loader)
    dataset       = image_libs.TransformationsDataset(images_aug)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    training = image_libs.KPPipeline(image_libs.KPModel, dataset, device)

    num_steps = 100000  

    for n in range(num_steps):
        result_log = training.step()

        if (n%100) == 0:
            print("step ", n)
            print(result_log)
            print(result_log["seg_metric"])
            print(result_log["kp_metric"])
            print("\n\n")

        if (n%(num_steps//10)) == 0:
            torch.save(training.model.state_dict(), "model.pt")
            print("saving model\n\n")  

    '''
    image_size = 512

    batch_size = 32

    x_in_batch, y_seg_batch, y_kp_batch = sample_batch(dataset, batch_size)
    x1_in_batch, y1_seg_batch, y1_kp_batch = sample_batch(dataset, batch_size)

    result = compute_segmentation_metrics(y_seg_batch, y1_seg_batch)

    print(result)
    '''