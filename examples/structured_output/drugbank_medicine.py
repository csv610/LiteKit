
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
import sys
import json

from litellm import completion


class DrugType(str, Enum):
    """Types of drugs"""
    SMALL_MOLECULE = "small_molecule"
    BIOTECH = "biotech"
    VACCINE = "vaccine"
    BIOLOGIC = "biologic"
    HERB = "herb"


class DrugGroup(str, Enum):
    """Drug approval groups"""
    APPROVED = "approved"
    EXPERIMENTAL = "experimental"
    INVESTIGATIONAL = "investigational"
    WITHDRAWN = "withdrawn"
    ILLICIT = "illicit"
    NUTRACEUTICAL = "nutraceutical"


class RouteOfAdministration(str, Enum):
    """Routes of drug administration"""
    ORAL = "oral"
    INTRAVENOUS = "intravenous"
    INTRAMUSCULAR = "intramuscular"
    SUBCUTANEOUS = "subcutaneous"
    TOPICAL = "topical"
    INHALATION = "inhalation"
    RECTAL = "rectal"
    TRANSDERMAL = "transdermal"
    OPHTHALMIC = "ophthalmic"
    NASAL = "nasal"
    SUBLINGUAL = "sublingual"
    OTHER = "other"


class ChemicalProperties(BaseModel):
    """Chemical properties of the medicine"""
    molecular_formula: Optional[str] = Field(None, description="Molecular formula")
    molecular_weight: Optional[float] = Field(None, description="Molecular weight in g/mol")
    smiles: Optional[str] = Field(None, description="SMILES notation")
    inchi: Optional[str] = Field(None, description="InChI identifier")
    inchi_key: Optional[str] = Field(None, description="InChI Key")
    iupac_name: Optional[str] = Field(None, description="IUPAC name")
    cas_number: Optional[str] = Field(None, description="CAS Registry Number")
    monoisotopic_weight: Optional[float] = Field(None, description="Monoisotopic weight")
    average_mass: Optional[float] = Field(None, description="Average mass")


class Taxonomy(BaseModel):
    """Drug taxonomy classification"""
    kingdom: Optional[str] = Field(None, description="Kingdom classification")
    superclass: Optional[str] = Field(None, description="Superclass")
    drug_class: Optional[str] = Field(None, description="Class")
    subclass: Optional[str] = Field(None, description="Subclass")
    direct_parent: Optional[str] = Field(None, description="Direct parent")
    alternative_parents: Optional[List[str]] = Field(default_factory=list, description="Alternative parents")


class ExternalIdentifier(BaseModel):
    """External database identifiers"""
    database: str = Field(..., description="Database name (e.g., PubChem, ChEMBL, KEGG)")
    identifier: str = Field(..., description="ID in that database")
    url: Optional[HttpUrl] = Field(None, description="Direct link to resource")


class Patent(BaseModel):
    """Patent information"""
    patent_number: str = Field(..., description="Patent number")
    country: str = Field(..., description="Country code")
    approved: Optional[datetime] = Field(None, description="Approval date")
    expires: Optional[datetime] = Field(None, description="Expiration date")
    pediatric_extension: Optional[bool] = Field(None, description="Has pediatric extension")


class ATCCode(BaseModel):
    """Anatomical Therapeutic Chemical Classification"""
    code: str = Field(..., description="ATC code")
    level: str = Field(..., description="ATC level description")


class Interaction(BaseModel):
    """Drug-drug interaction"""
    drug_name: str = Field(..., description="Interacting drug name")
    drugbank_id: Optional[str] = Field(None, description="DrugBank ID of interacting drug")
    description: str = Field(..., description="Interaction description")
    severity: Optional[str] = Field(None, description="Severity level (major, moderate, minor)")


class FoodInteraction(BaseModel):
    """Drug-food interaction"""
    food: str = Field(..., description="Food or nutrient")
    description: str = Field(..., description="Interaction description")


class Target(BaseModel):
    """Biological target (protein, enzyme, receptor)"""
    target_id: str = Field(..., description="Target ID")
    name: str = Field(..., description="Target name")
    organism: str = Field(..., description="Organism")
    action: Optional[str] = Field(None, description="Type of action (inhibitor, agonist, etc.)")
    gene_name: Optional[str] = Field(None, description="Gene name")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID")
    pharmacological_action: Optional[bool] = Field(None, description="Is pharmacologically active")


class Enzyme(BaseModel):
    """Enzyme involved in drug metabolism"""
    enzyme_id: str = Field(..., description="Enzyme ID")
    name: str = Field(..., description="Enzyme name")
    organism: str = Field(..., description="Organism")
    action: Optional[str] = Field(None, description="Metabolic action")
    gene_name: Optional[str] = Field(None, description="Gene name")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID")


class Carrier(BaseModel):
    """Carrier protein"""
    carrier_id: str = Field(..., description="Carrier ID")
    name: str = Field(..., description="Carrier name")
    organism: str = Field(..., description="Organism")
    gene_name: Optional[str] = Field(None, description="Gene name")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID")


class Transporter(BaseModel):
    """Transporter protein"""
    transporter_id: str = Field(..., description="Transporter ID")
    name: str = Field(..., description="Transporter name")
    organism: str = Field(..., description="Organism")
    action: Optional[str] = Field(None, description="Transport action")
    gene_name: Optional[str] = Field(None, description="Gene name")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID")


class Pharmacodynamics(BaseModel):
    """Pharmacodynamic properties"""
    mechanism_of_action: Optional[str] = Field(None, description="Detailed mechanism of action")
    pharmacodynamics: Optional[str] = Field(None, description="Pharmacodynamic description")
    onset_of_action: Optional[str] = Field(None, description="Time to onset")
    duration_of_action: Optional[str] = Field(None, description="Duration of effect")
    peak_effect: Optional[str] = Field(None, description="Time to peak effect")


class Pharmacokinetics(BaseModel):
    """Pharmacokinetic properties"""
    absorption: Optional[str] = Field(None, description="Absorption characteristics")
    distribution: Optional[str] = Field(None, description="Distribution characteristics")
    volume_of_distribution: Optional[str] = Field(None, description="Volume of distribution")
    protein_binding: Optional[str] = Field(None, description="Protein binding percentage")
    metabolism: Optional[str] = Field(None, description="Metabolism description")
    route_of_elimination: Optional[str] = Field(None, description="Route of elimination")
    half_life: Optional[str] = Field(None, description="Elimination half-life")
    clearance: Optional[str] = Field(None, description="Clearance rate")
    bioavailability: Optional[str] = Field(None, description="Bioavailability")


class Dosage(BaseModel):
    """Dosage information"""
    form: str = Field(..., description="Dosage form (tablet, capsule, injection, etc.)")
    route: RouteOfAdministration = Field(..., description="Route of administration")
    strength: str = Field(..., description="Strength/concentration")
    dosage_instructions: Optional[str] = Field(None, description="Detailed dosing instructions")


class ClinicalTrial(BaseModel):
    """Clinical trial information"""
    trial_id: str = Field(..., description="Trial ID (e.g., NCT number)")
    title: str = Field(..., description="Trial title")
    phase: Optional[str] = Field(None, description="Trial phase")
    status: Optional[str] = Field(None, description="Trial status")
    url: Optional[HttpUrl] = Field(None, description="Link to trial information")


class Manufacturer(BaseModel):
    """Drug manufacturer information"""
    name: str = Field(..., description="Manufacturer name")
    country: Optional[str] = Field(None, description="Country")
    url: Optional[str] = Field(None, description="Company website")


class PricingInfo(BaseModel):
    """Pricing information by country"""
    country: str = Field(..., description="Country code or name")
    price: str = Field(..., description="Price information")


class Contraindication(BaseModel):
    """Contraindication information"""
    condition: str = Field(..., description="Contraindicated condition")
    severity: Optional[str] = Field(None, description="Severity (absolute, relative)")
    description: Optional[str] = Field(None, description="Detailed description")


class AdverseReaction(BaseModel):
    """Adverse drug reaction"""
    reaction: str = Field(..., description="Adverse reaction")
    frequency: Optional[str] = Field(None, description="Frequency (common, rare, etc.)")
    severity: Optional[str] = Field(None, description="Severity level")

class BasicInfo(BaseModel):
    drugbank_id: Optional[str] = Field(None, description="DrugBank accession ID")
    name: str = Field(..., description="Primary drug name")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names and brand names")
    description: Optional[str] = Field(None, description="Comprehensive drug description")
    chemical_properties: Optional[ChemicalProperties] = Field(None, description="Chemical properties")
    affected_organisms: Optional[List[str]] = Field(default_factory=list, description="Affected organisms")
    synthesis_reference: Optional[str] = Field(None, description="Chemical synthesis information")

class Classification(BaseModel):
    drug_type: Optional[DrugType] = Field(None, description="Type of drug")
    groups: List[DrugGroup] = Field(default_factory=list, description="Drug approval groups")
    categories: List[str] = Field(default_factory=list, description="Therapeutic categories")
    atc_codes: List[ATCCode] = Field(default_factory=list, description="ATC classification codes")
    taxonomy: Optional[Taxonomy] = Field(None, description="Chemical taxonomy")

class Pharmacology(BaseModel):
    pharmacodynamics: Optional[Pharmacodynamics] = Field(None, description="Pharmacodynamic properties")
    pharmacokinetics: Optional[Pharmacokinetics] = Field(None, description="Pharmacokinetic properties")

class Indications(BaseModel):
    indication: Optional[str] = Field(None, description="Approved indications")
    off_label_uses: Optional[List[str]] = Field(default_factory=list, description="Off-label uses")

class Administration(BaseModel):
    dosages: List[Dosage] = Field(default_factory=list, description="Available dosage forms")
    administration: Optional[str] = Field(None, description="Administration instructions")
    overdose: Optional[str] = Field(None, description="Overdose information")

class Interactions(BaseModel):
    targets: List[Target] = Field(default_factory=list, description="Biological targets")
    enzymes: List[Enzyme] = Field(default_factory=list, description="Metabolizing enzymes")
    carriers: List[Carrier] = Field(default_factory=list, description="Carrier proteins")
    transporters: List[Transporter] = Field(default_factory=list, description="Transporter proteins")
    drug_interactions: List[Interaction] = Field(default_factory=list, description="Drug-drug interactions")
    food_interactions: List[FoodInteraction] = Field(default_factory=list, description="Drug-food interactions")

class Safety(BaseModel):
    contraindications: List[Contraindication] = Field(default_factory=list, description="Contraindications")
    warnings: Optional[str] = Field(None, description="Warnings and precautions")
    adverse_reactions: List[AdverseReaction] = Field(default_factory=list, description="Adverse reactions")
    black_box_warning: Optional[str] = Field(None, description="FDA black box warning")
    pregnancy_category: Optional[str] = Field(None, description="Pregnancy category")
    lactation: Optional[str] = Field(None, description="Lactation information")
    toxicity: Optional[str] = Field(None, description="Toxicity information")

class Regulation(BaseModel):
    approval_date: Optional[datetime] = Field(None, description="FDA approval date")
    patents: List[Patent] = Field(default_factory=list, description="Patent information")
    manufacturers: List[Manufacturer] = Field(default_factory=list, description="Manufacturers")
    pricing: List[PricingInfo] = Field(default_factory=list, description="Pricing information by country")

class References(BaseModel):
    external_identifiers: List[ExternalIdentifier] = Field(default_factory=list, description="External database IDs")
    clinical_trials: List[ClinicalTrial] = Field(default_factory=list, description="Associated clinical trials")
    literature_references: Optional[List[str]] = Field(default_factory=list, description="PubMed IDs and references")

class MedicineInfo(BaseModel):
    basic_info: BasicInfo = Field(..., description="Basic medicine information")
    classification: Classification = Field(None, description="Drug classification")
    pharmacology: Optional[Pharmacology] = Field(None, description="Pharmacology information")
    indications: Optional[Indications] = Field(None, description="Indications for use")
    administration: Optional[Administration] = Field(None, description="Administration guidelines")
    interactions: Optional[Interactions] = Field(None, description="Drug interactions and targets")
    safety: Optional[Safety] = Field(None, description="Safety information")
    regulation: Optional[Regulation] = Field(None, description="Regulatory information")
    references: Optional[References] = Field(None, description="External references and links")

def cli(medicine):
    model = "gemini/gemini-2.5-flash"

    # Update the user message to request information about a specific medicine
    messages = [{"role": "user", "content": f"Provide detailed information about the medicine {medicine}."}]

    response = completion(
        model=model,
        messages=messages,
        response_format=MedicineInfo
    )

    # Parse and print the formatted JSON output
    json_string = response.choices[0].message.content
    data = json.loads(json_string)

    output_filename = f"{medicine}.json"
    with open(output_filename, "w") as f:
        f.write(json.dumps(data, indent=4))
    print(f"Medicine information saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python drugbank_medicine.py <medicine_name>")
        sys.exit(1)
    medicine = sys.argv[1]
    cli(medicine)

