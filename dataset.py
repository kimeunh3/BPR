import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
            self,
            data: Dataset,
            idx: list,
            config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.data = data
    def __len__(self) -> int:
        """
        return data length
        """
        return len(self.user_list)

    def __getitem__(self, index: int) -> object:
        
        return {"user": , "": num, "answerCode": y, "mask": mask}

