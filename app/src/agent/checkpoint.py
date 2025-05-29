from typing import Any, Dict, Optional, cast
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver

class CustomMemorySaver(MemorySaver):
    """Custom memory saver that handles non-picklable objects."""
    
    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        # Create a shallow copy of the checkpoint
        checkpoint_dict = checkpoint.copy()
        
        # Handle the state specifically
        if "state" in checkpoint_dict and "messages" in checkpoint_dict["state"]:
            # Create a new messages list without non-picklable objects
            new_messages = []
            for msg in checkpoint_dict["state"]["messages"]:
                # Create a new message dict without non-picklable attributes
                msg_dict = msg.dict()
                # Remove any attributes that might contain non-picklable objects
                msg_dict.pop("additional_kwargs", None)
                new_messages.append(msg_dict)
            
            # Update the checkpoint with the cleaned messages
            checkpoint_dict["state"]["messages"] = new_messages
        
        # Call the parent's aput with the cleaned checkpoint
        return await super().aput(config, checkpoint_dict, metadata)

def get_checkpoint_saver():
    return CustomMemorySaver()
