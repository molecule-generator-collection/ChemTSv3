import xml.etree.ElementTree as ET

class MonomersLib():
    def __init__(self, monomers_lib: ET):
        self.lib = monomers_lib
        self.__class__.strip_namespace(monomers_lib.getroot())
    
    @classmethod
    def load(cls, monomer_lib_path: str):
        monomers_lib = ET.parse(monomer_lib_path)
        return cls(monomers_lib)
    
    # remove namespace from xml
    @classmethod
    def strip_namespace(cls, element: ET.Element):
        element.tag = element.tag.split("}", 1)[-1] if "}" in element.tag else element.tag
        for child in element:
            cls.strip_namespace(child)
    
    # load monomers of the specific polymer type
    def get_monomers_list(self, polymer_type: str):
        root = self.lib.getroot()
        polymer_list = root.find("PolymerList")
        monomers = polymer_list.find("Polymer[@polymerType='" + polymer_type + "']")
        return monomers

    def get_monomer(self, polymer_type: str, monomer_id: str):
        monomers_list = self.get_monomers_list(polymer_type)
        monomer = monomers_list.find("Monomer[MonomerID='" + monomer_id + "']")
        return monomer

    def get_monomer_smiles(self, polymer_type: str, monomer_id: str):
        monomer = self.get_monomer(polymer_type, monomer_id)
        return monomer.find("MonomerSmiles").text
    
    def get_attachment_cap_smiles(self, polymer_type: str, monomer_id: str, attachment_label: str):
        return self.get_monomer(polymer_type, monomer_id).find("Attachments/Attachment[AttachmentLabel='" + attachment_label + "']/CapGroupSmiles").text
