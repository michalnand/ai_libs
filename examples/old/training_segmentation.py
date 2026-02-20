import image_libs
import numpy
import torch


if __name__ == "__main__":

    
    #dataset_path = "/Users/michal/datasets/rugd/train/"
    dataset_path = "/home/michal/datasets/rugd/train/"  

    images_loader      = image_libs.ImagesLoader(dataset_path + "/img/")
    annotations_loader = image_libs.RUGDAnnLoader(dataset_path + "/ann/")

    dataset = image_libs.SegmentationDataset(images_loader, annotations_loader, image_size = (256, 512))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    training = image_libs.SegmentationPipeline(image_libs.SegModel, dataset, device)

    num_steps = 100000  

    for n in range(num_steps):
        result_log = training.step()

        if (n%100) == 0:
            print("step ", n)
            print(result_log)
            print(result_log["seg_metric"])
            print("\n\n")

        if (n%(num_steps//10)) == 0:
            torch.save(training.model.state_dict(), "model_seg.pt")
            print("saving model\n\n")  
