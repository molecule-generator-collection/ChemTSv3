from .base import Node, MolNode, SurrogateNode
from .sentence_node import SentenceNode, MolSentenceNode
from .string_node import StringNode, MolStringNode, SMILESStringNode, CanonicalSMILESStringNode, FASTAStringNode

# lazy import
def __getattr__(name):
    if name == "SELFIESStringNode":
        from .selfies_string_node import SELFIESStringNode
        return SELFIESStringNode
    raise AttributeError(f"module {__name__} has no attribute {name}")