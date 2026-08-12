"""
Co-location hybrid-project sourcing (2026-08-12): real, primary-source content
specifically about combining BESS with wind/solar/hydro generation at ONE
shared grid connection point -- shared capacity allocation, dual-permit
categories, dual-land-use planning privilege. This is the sourcing gap
identified during the hybridi coverage investigation: the hybridi-tag-
inheritance rule (applied across PR-TAG-1..7c) correctly scopes which
EXISTING single-technology documents are relevant background reading for a
hybrid project, but no source anywhere in the corpus actually discussed the
intersection itself. These 5 documents fill that gap for 4 of the 9
countries (FI, DE x2, DA, LV) where real content was found and verified by
direct primary-source retrieval; LT stayed blocked (vert.lt WAF 403,
consistent with the existing LT bot-blocked backlog -- see the manual-
sourcing-backlog memory) and is not included here.

Content-quality notes per source, and how each was actually retrieved (not
reconstructed from memory) are in each DOCS entry's own comment below.

hanketyyppi_tag is deliberately NOT set in this ingestion's metadata -- same
choice as ingest_maatalous_vesivoima.py, for the same reason: resolved
exclusively via source_policy.py's SOURCE_HANKETYYPPI_TAG (single source of
truth), not duplicated/baked into chunk metadata. This was a hard-won lesson
from the LV/LT hybridi backfill PR immediately before this one -- baking the
tag into metadata at ingest time means future dict-only corrections silently
stop working without a live ChromaDB migration. Avoiding that here entirely.

Kaytto:
    python3 permit_ai/ingest_hybridi_colocation.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DB_DIR = HERE / "embeddings"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION  = "permit_docs"
CHUNK_CHARS = 1500
OVERLAP     = 200
BATCH       = 32

# doc_id -> (source_label, url, country, lang, text)
DOCS: list[tuple[str, str, str, str, str, str]] = [
    (
        "fi_fingrid_hybridivoimalaitos_ohje_2023",
        "Fingrid, Instruction / Application of the grid code specifications to hybrid power plants, 17.10.2023 (unofficial translation)",
        "https://www.fingrid.fi/globalassets/dokumentit/en/customers/grid-connection/instruction---application-of-the-grid-code-specifications-to-hybrid-power-plants---2023_10_17.pdf",
        "FI",
        "en",
        # Retrieved via direct PDF fetch + pypdf text extraction (WebFetch's
        # summarizer could not parse this PDF's encoding; raw text extracted
        # directly instead). Full 23-page document, real primary source --
        # not a summary. Formally defines "hybrid power plant" (multiple
        # plant sections of different types -- solar, wind, hydro, or grid
        # energy storage -- behind one connection point), and gives 3 real
        # worked numeric examples of combined-capacity/reactive-power/fault-
        # ride-through sharing: 100MW wind + 50MW solar + 20MW storage at a
        # 130MW connection; 30MW wind + 15MW storage at a 30MW connection;
        # 2x1.2MW storage added to an existing 2x20MW hydropower plant.
        """\
 
      UNOFFICIAL TRANSLATION 
Instruction / Application of the grid code specifications to hybrid power plants 
17.10.2023 
  

 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 2 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
Table of contents 
1 Introduction ............................................................................................................................................ 4 
1.1 Compliance monitoring process and operational notification procedure  ......................................... 4 
1.2 Modifications to existing plants ........................................................................................................ 5 
2 Dimensioning values of hybrid power plants ..................................................................................... 5 
2.1 Rated capacity and minimum output ............................................................................................... 5 
2.2 Reactive power capacity .................................................................................................................. 6 
2.3 Fault ride-through ............................................................................................................................ 9 
2.4 Voltage control ................................................................................................................................. 9 
2.5 Protection and fault current injection ............................................................................................... 9 
2.6 Measurements and remote control ................................................................................................ 10 
2.7 Other technical requirements ........................................................................................................ 11 
3 Technical data and simulation models to be provided for hybrid power plants  .......................... 12 
3.1 Information to be provided ............................................................................................................. 12 
3.2 Modelling requirements ................................................................................................................. 12 
4 Commissioning tests for hybrid power plants ................................................................................. 13 
4.1 Commissioning tests...................................................................................................................... 14 
4.2 Monitoring period ........................................................................................................................... 16 
5 References ........................................................................................................................................... 17 
Appendix 1 Application examples of hybrid power plants .................................................................. 18 
1 Example 1 ............................................................................................................................................. 18 
2 Example 2 ............................................................................................................................................. 20 
3 Example 3 ............................................................................................................................................. 22 
 
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 3 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
Change history 
Date Version Change 
17 October 
2023 1.0 First version 
   
   
 
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 4 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
1  Introduction 
These instructions describe how the VJV2018 /1/ and SJV2019 /2/ grid code 
specifications should apply to hybrid power plants.  
A hybrid power plant is a power plant where plant sections of different types are 
connected to a single connection point. For example, power plants with different primary 
energy sources (solar, wind, hydro) or grid energy storage systems with active or reactive 
power controlled by a central controller may be connected to one connection point.  
A central controller is a controller that makes the operation of the plant sections 
dependent on each other. The following are not defined as central controllers: 
• Controlling the tap-changer of a main transformer shared between the various 
sections of a hybrid power plant 
• Slow upper-level intra-plant reactive power control (see VJV2018, Appendix B, 
section 22.4), which is only permitted for connections with both production and 
consumption 
If the power plant sections operate independently of each other and are controlled by 
dedicated controllers, the sections are considered independent power plants. Examples 
of independent power plants include conventional hydroelectric power production units 
where the turbines have dedicated water routes and voltage control based on the 
terminal voltage of their generators. 
The physical locations of plant sections in relation to one another are irrelevant to the 
definition of a hybrid power plant. 
These instructions only apply to hybrid power plants.   
1.1  Compliance monitoring process and operational notification procedure 
The compliance monitoring process and operational notification procedure in VJV2018 
apply to hybrid power plants.  
If a hybrid power plant project is constructed in phases, the connecting party must agree 
with Fingrid and the network operator on the phasing of the VJV grid code compliance 
monitoring process and operational notification procedure, taking into account the 
deadlines set in the VJV for the validity of the interim operational notification (ION). In 
principle, every plant section should be completed without delay, avoiding unnecessary 
extensions to the compliance monitoring process. Between the construction phases, 
Fingrid must have access to the latest technical and modelling data that reliably describe 
the plant’s operation.  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 5 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
1.2  Modifications to existing plants 
Converting an existing power plant into a hybrid power plant requires the initiation of a 
VJV compliance monitoring process in accordance with VJV2018 to re-examine the 
individual plant sections as required and assess the compliance of the hybrid power plant 
entity. The starting point for designing a hybrid power plant is to ensure the uniform 
operation of all plant sections in line with the existing VJV2018 requirements.  
In principle, the applicable grid code specifications apply to plant modifications. If an 
existing plant section is designed to fulfil the grid code specifications that applied before 
VJV2018, the connecting party must evaluate the plant section’s capability of meeting the 
VJV2018 requirements and endeavour to adapt its operation accordingly. If an existing 
plant section requires technically and financially significant modifications, the connecting 
party may ask Fingrid to limit the scope of the modifications. Fingrid decides whether to 
limit the scope of modifications on a case-by-case basis, provided that the modifications 
do not prevent new plant sections from complying with the requirements.  
2  Dimensioning values of hybrid power plants 
2.1  Rated capacity and minimum output 
The type class (A-D) of a hybrid power plant and corresponding technical requirements 
are based on the hybrid power plant’s rated capacity and the connection point’s voltage 
level in accordance with the classification in VJV2018 and SJV2019. For example, a type 
D hybrid power plant is defined as an entity with a rated capacity of at least 30 MW or a 
connection point voltage level of at least 110 kV.    
The rated capacity of a hybrid power plant (Pmax) is its highest active power production 
level measured at the connection point, as specified in the connection agreement or 
otherwise determined between the network operator and the connecting party. The rated 
capacity of a hybrid power plant must be at least as high as the active power of the 
largest plant section without any software based limitations on the active power. The 
rated capacity of a hybrid power plant may be no higher than the combined active power 
of the rated capacities of each plant section (Pmax 1–Pmax n). The hybrid power plant’s rated 
capacity and the rated capacities of the individual plant section must always be agreed 
upon with the network operator at the connection point, and the agreed capacities must 
not be exceeded.  
The minimum output of a hybrid power plant (Pmin) is based on the hybrid power plant’s 
largest plant section. It can be no higher than 10% of the rated active power in 
accordance with section 16.3.2.1 of VJV2018. If the plant sections are equal in size, the 
hybrid power plant’s minimum output is determined according to the plant section with the 
highest minimum output. In addition, the minimum output must be determined for each 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 6 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
plant section (Pmin 1–Pmin n) based on their actual technical performance. No minimum 
output is defined for grid energy storage systems. 
In 110 kV and 400 kV switchgear connections to Fingrid’s network and 110 kV 
transmission line connections along a Fingrid transmission lines built in accordance with 
the general connection terms YLE2021, a hybrid power plant’s rated capacity (Pmax) may 
be limited by software. The rated capacity is not based on the sum of the rated capacities 
of plant sections. In line with YLE2021, a hybrid power plant’s largest stepwise power 
change at the power plant connection may not exceed 1,300 MW.    
2.2  Reactive power capacity 
A hybrid power plant’s reactive power capacity requirement is determined according to 
the plant’s type class. The requirement is valid in full when the hybrid plant’s largest 
section is operating above its minimum output. The reactive power capacities of other 
plant sections allocated for voltage control and with readiness to produce must not be 
restricted by software below the actual technical capability of the equipment while 
operating below this minimum output level.  
If individual plant sections can also operate independently (for example, when the other 
plant sections are not operating), they must then meet the reactive power capacity 
requirement according to the plant section’s rated capacity at the point where the hybrid 
power plant’s reactive power capacity requirement is defined.  
The reactive power capacity requirement for type C and D hybrid power plants 
corresponds to sections 12.2.2 and 17.2.1 of VJV2018 (Figure 1). If a hybrid power plant 
includes a type C or D grid energy storage system, the reactive power capacity 
requirement also applies to this plant section when in consumption mode (Figure 2). If a 
plant section has less than 10% of the main transformer’s nominal power, the relevant 
network operator at the connection point will specify the reactive power capacity 
requirement for the specific plant section on a case-by-case basis. 
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 7 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
Figure 1. The VJV2018 reactive power capacity requirement for type C and D power 
generating facilities. 
 
 
Figure 2. The SJV2019 reactive power capacity requirement for type C and D grid energy 
storage systems. 
  
Under-
excited 
Under-
excited 
Over-
excited 
Under-
excited 
Over-
excited 
(production) 
(consumption) 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 8 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
In accordance with VJV2018 and SJV2019, the hybrid power plant’s reactive power 
capacity requirement must be met at the power plant’s connection point. In line with 
Fingrid’s prior interpretation – see /3/ chapter 2 – the terminals on the high voltage side of 
a hybrid power plant’s main transformer (or a busbar shared between several main 
transformers) may be used as the point at which a hybrid power plant’s reactive power 
capacity requirement is determined instead of the connection point. In such a case, the 
hybrid power plant’s imputed rated capacity will be the highest active power defined to 
this point. Connection networks and associated losses between the main transformer and 
the connection point do not affect the hybrid power plant’s rated capacity or the 
determination of the reactive power capacity. However, additional reactive power capacity 
may be required if a power plant has a very long connecting line (section 17.2.2 of 
VJV2018). This should be agreed upon separately with the network operator at the 
connection point.   
The reactive power capacity requirement for a hybrid power plant and plant sections 
operating independently can be met using a combination of reactive power capacities 
from plant sections contributing to voltage control. The reactive power capacities of plant 
sections contributing to voltage control must not be needlessly limited by software. 
Dimensioning should take into account the functional constraints of the plant sections, 
including the following:  
• Minimum output (e.g., the reactive power production capacity at zero active 
power) 
• Sections causing constraints (e.g., the capacity of a shared main transformer)  
• Operational constraints (e.g., availability at specific times) 
The reactive power capacity should primarily be dynamic. In other words, the reactive 
power capacity should be implemented with converters offering rapid, step-free control. 
Note that switchable additional compensation, such as mechanically switchable 
capacitors, do not count as dynamic. If the total reactive power capacity of a hybrid power 
plant’s production-ready (connected) sections without any software limitations is 
insufficient to meet the reactive power capacity requirement based on the rated capacity, 
up to 15% of it may be covered by switchable additional compensation. However, the 
hybrid power plant shall be capable of meeting the reactive power capacity required in full 
without switchable additional compensation when the power plant’s active power output is 
less than 85% of the rated capacity (Pmax). Instruction /3/ provides more detailed guidance 
on the operation and dimensioning of additional compensation. 
If a hybrid power plant is incapable of meeting its reactive power capacity requirement 
according to the rated capacity for its operating state at any given time – for example, due 
to the technical failure of an individual plant section – the power plant’s active power 
should be limited to a level corresponding to the reactive power capacity at the time (for 
type C and D plants, Pmax≤|Qmax/0.33|, taking into account the voltage conditions).  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 9 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
If a hybrid power plant has a very high reactive power capacity, it may be necessary to 
use software to limit the reactive power capacity at Fingrid’s request. Even then, the 
reactive power capacity requirement conforms to the VJV/SJV requirements.  
2.3  Fault ride-through 
A hybrid power plant’s fault ride-through requirement is determined according to its type 
class at the hybrid power plant’s connection point.  
If individual plant sections can also operate independently (for example, when the other 
plant sections are not operating), they must then meet the fault ride-through requirement 
according to the plant section’s type class at the hybrid power plant’s connection point.  
2.4  Voltage control 
According to /4/, the primary control method for all power plants with a power of over 
10 MW or a connection point voltage level of at least 110 kV (type C and D power plants) 
is constant voltage control. This also applies to hybrid power plants. If a hybrid power 
plant consists of several plant sections of less than 10 MW, but its rated capacity exceeds 
10 MW, or the voltage at its connection point is at least 110 kV, it must operate under 
continuous voltage control.  
All the plant sections contributing to meeting the reactive power capacity requirement of a 
type C or D hybrid power plant must operate under continuous voltage control. The rated 
reactive power (Qn) used to determine the voltage control slope (VJV2018, section 
22.3.1) is based on the hybrid power plant’s rated capacity (Qn = 0.33 x Pmax). If plant 
sections operate independently, the rated reactive power is based on each plant section’s 
rated capacity (Qn1 = 0.33 x Pmax1 etc.)  
In principle, the voltage control point (reference point) for the entire hybrid power plant is 
at the high-voltage side of the power plant’s main transformer, which is typically at the 
power plant’s 110 kV busbar. Voltage control can also be implemented for specific plant 
sections (the plant sections control the same busbar voltage with the same voltage slope 
configuration) while taking into account the capability to manage the total reactive power 
at the connection point. This avoids overloading the common plant-level components, 
such as the main transformer.  
The lower-level controls of specific plant sections shall be coordinated with each other 
and the upper plant-level control so that voltage control functions stably under normal 
operating conditions and disturbances and no harmful interaction phenomena arise. 
2.5  Protection and fault current injection 
Hybrid power plants are not permitted to supply active power in excess of their rated 
capacity into the grid. Only short-term and attenuating power fluctuations in excess of the 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 10 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
rated capacity are permitted when caused by the power plant’s dynamic response to 
transient phenomena in the power system. If the active power simultaneously available 
from a hybrid power plant’s sections may exceed the plant’s rated capacity when not 
restricted by the controller, the hybrid power plant must be equipped with a protective 
device to ensure the plant does not exceed its rated capacity. The protective device must 
measure the power plant’s active power and disconnect the power plant or an individual 
plant section if the power exceeds 105% x Pmax for 20 seconds. The total power 
measured at the high-voltage terminals of the power plant’s main transformers may be 
used as the active power measured by the protective device if no measurements are 
available at the actual connection point. The protective function can be realised as part of 
an existing protection relay (separate protection function).  
If the plant is connected to a 110 kV Fingrid transmission line, the fault current injection of 
the plant sections must be taken into account and limited according to YLE2021 section 
2.5 to 1.2 times the nominal current based on the hybrid power plant’s rated capacity 
(300 ms from the onset of the fault). The number of converters connected in each 
operating state must be taken into account in fault current injection.   
If the plant is connected by switchgear to a Fingrid substation, the fault current injection of 
plant sections must not be limited unless separately agreed upon with Fingrid.  
2.6  Measurements and remote control 
Hybrid power plants must provide the following real-time measurements: 
• Active and reactive power measurements for each plant section and total powers 
for the plant as a whole. 
• Switching device position indication in a scope specified on a case-by-case basis 
based on the plant’s single line diagram. In principle, the plant should send status 
data on the (high-voltage) circuit breaker, disconnector and earthing switch from 
the power plant’s substation to the grid, as well as the main circuit breakers in 
each plant section.   
• Voltage measurement from the busbar that the plant uses to control the voltage 
when operating at constant voltage control. This applies to types C and D. 
• Status information of the power plant’s plant-level control state (voltage 
control/reactive power control/power factor control). This applies to types C and D. 
A type C or D hybrid power plant must also be capable of receiving maximum power 
orders as electronic remote control commands from Fingrid and acknowledging receipt of 
the information and compliance with the order. Electronic remote control will be 
implemented between the SCADA systems of Fingrid and the operator responsible for 
the hybrid power plant’s operation. 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 11 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
Fingrid’s real-time information exchange application instructions provide more detailed 
requirements for real-time data exchange /6/. 
All the plant sections in a hybrid power plant must always be controllable with a 15-
minute response time in accordance with VJV2018 section 10.4.1. If the operation of the 
plant is based on remote operation (remote control) from the control centre of the entity 
responsible for the power plant’s operation, as authorised by the connecting party, the 
performance of the remote control functions must be verified during the commissioning 
phase, as soon as the plant sections begin supplying power to the connection point.  
Fingrid recommends equipping hybrid power plants with continuously operating data 
loggers that provide the operator responsible for the power plant’s operation with 
immediate access to the measurements. The data logger should measure the currents 
and voltages at the connection point with a high sampling rate (> 5 kHz). A 30-day 
memory capacity is also recommended. The purpose of the data logger is to enable the 
operation of the hybrid power plant to be analysed in its normal operating state and in the 
event of disturbances and changes in the power system. 
2.7  Other technical requirements 
If the control of a converter-connected plant section is based on Grid Forming (GFM) 
control, the technical operating principles of the installation should be agreed upon with 
Fingrid on a case-by-case basis. The voltage control functionality of such a plant section 
must be carefully coordinated with other converter-connected plant sections - which are 
typically operating in the Grid Following (GFL) control - by taking into account the 
interactions between individual controllers and plant-level controls. The design must 
consider the following: 
• Fault current injection and restoration from faults 
• The implementation of voltage control and active power control. Plant-level 
controls must not significantly restrict GFM control from responding rapidly to 
change phenomena in the grid.   
• When installations based on GFM control are installed on 110 kV transmission 
line connections, they must be prevented from unintentionally entering island 
operation by implementing a disconnection datalink to the hybrid power plant if the 
line does not have a protection datalink. 
• Switching to house load operation, synchronization to the grid and possible 
blackstart capability. 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 12 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
3  Technical data and simulation models to be provided for 
hybrid power plants 
3.1  Information to be provided 
A PSS/E simulation model, model documentation, plant documentation, calculations 
(voltage control performance, reactive power capacity and voltage disturbance 
calculations) and the material required due to any specific study requirements (VJV2018, 
section 5) set for the project must be provided for hybrid power plants of type C or D 
(VJV2018 and SJV2019) in accordance with the VJV, the SJV (if applicable) and Fingrid’s 
modelling instructions /5/. A PSCAD simulation model and model documentation must 
also be provided for type D hybrid power plants. 
The plant documentation must contain a detailed system description of the hybrid power 
plant’s operating principles, such as the following: 
• The operating modes of plant sections: active power supply constraints and the 
technical background for them, the order of precedence of the production forms, 
an estimate of each plant section’s annual production (power, energy and 
distribution over the hours of the year) 
• The implementation of reactive power capacity, taking into account the 
instantaneous availability of plant sections (may be presented as part of the 
reactive power capacity calculation) 
• Whether power converters are always online, irrespective of the readiness for 
active power production (for example, the STATCOM feature of wind turbine 
converters, “Night mode” in grid energy storage power converters when operating 
without batteries, operating mode of solar power plant power converters in the 
winter) 
The information referred to above should be submitted to the relevant network operator 
for review at least 6 months before the power plant is commissioned, that is, when the 
plant is intended to begin supplying power to the grid. 
3.2  Modelling requirements 
Fingrid must have simulation models that describe the power plant’s true operation in 
adequate detail and conform to the requirements before the power plant begins supplying 
electricity to the grid. The simulation models provided for hybrid power plants must 
include all the plant sections and equipment used to control them, such as the central 
controller. If the model consists of several separate models, the connecting party is 
responsible for the functional coordination of the models.  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 13 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
When the connecting party plans its project schedule, it is important to remember that 
reviewing and correcting the model is an iterative process that typically takes several 
months. Hybrid power plants are very challenging to model, considering the multiple 
supply limits and technical complexity due to the implementation of control. Fingrid 
checks the provided models with the network operator at the connection point. The 
simulation models must be approved before an Interim Operational Notification (ION) can 
be issued.  
PSS/E and PSCAD models do not need to be submitted for grid energy storage systems 
installed to synchronous machine power plants to balance the frequency control of the 
plant if the rated capacity of the energy storage system is less than 5 MW and it is 
incapable of operating independently. 
The voltage disturbance calculation, voltage control performance calculation and the 
calculations required by the specific study requirements shall be prepared separately for 
all the planned operating states of the hybrid power plant – meaning when the plant 
sections operate together and independently – and submitted as a report. Fingrid verifies 
the other functionalities required of the models by VJV/SJV (such as power control) as 
part of its model review, and the connecting party does not need to report them.  
The modelling data for type C and D hybrid power plants is verified based on the results 
of commissioning tests with respect to the plant’s reactive power capacity and the 
operation of the related restriction controls. Section 4.2 describes the verification related 
to the monitoring period included in the commissioning tests. The modelling data for type 
D hybrid power plants shall also be verified with respect to any special regulation 
functionality tested during commissioning. If a hybrid power plant’s voltage control is 
tuned according to the VCSCR value, Fingrid will validate the model against the voltage 
control performance results obtained from commissioning tests and this is not required 
from the connecting party.  
4  Commissioning tests for hybrid power plants 
The reactive power capacity tests required by VJV2018 and SJV2019 for type C and D 
plants must be performed at 60% or more of the rated capacity. However, it may be 
difficult to arrange the simultaneous operation of the hybrid power plant sections at high 
power in practice. Moreover, the rated capacity agreed for the connection point may not 
even permit the plant sections to operate simultaneously at such high power. For this 
reason, the hybrid power plant’s reactive power capacity can be verified for each plant 
section individually according to the VJV/SJV requirements. The central controller’s 
allocation of reactive power between the plant sections is then verified by a separate test 
performed at a lower power level in accordance with Table 1.   
During the commissioning tests, at least 90% of the tested plant section’s production 
units must be available in the test and operate normally.  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 14 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
If a new plant section is added to a hybrid power plant, all the commissioning tests 
should, in principle, be repeated to the full extent. If it is possible to show that the addition 
of the new plant section will not affect the performance of the existing plant sections in a 
certain area, there is no need to repeat the commissioning test for the said area.  
4.1  Commissioning tests 
Table 1 shows the minimum scope of commissioning tests for type C and D hybrid power 
plants. The scope of testing is supplemented when necessary, for example, with testing 
of the special control functions of a specific hybrid plant. If the plant sections are also 
capable of independent operation under the control of a separate section-specific 
controller, commissioning tests must be performed on them in this operating state in the 
scope required by their type class under VJV/SJV.  
Table 1. Commissioning tests for type C and D hybrid power plants. 
Commissioning test Entire hybrid plant 
(all plant sections 
operating under the 
control of the central 
controller) 
Test each plant 
section separately 
1 Limited frequency 
sensitive mode – over-
frequency (LFSM-O) 
Yes No 
2 Limited frequency 
sensitive mode – under-
frequency (LFSM-U) 
Yes No 
3 Frequency sensitive 
mode (FSM) 
Yes No 
4 Rate of change in active 
power 
Yes Yes 
5 Constant voltage Yes Yes 
6 Control reactive power 
control 
Yes Yes 
7 Constant power factor 
control 
Yes Yes 
8 Reactive power 
capacity test and 
Yes. Test when all 
plant sections are 
operating 
simultaneously at a 
Yes 
The test is performed 
at all three power 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 15 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
restriction of active 
power 
minimum power of 
20%.  
The test should also 
verify the possible 
substitution of the plant 
sections’ reactive 
power capacity with 
capacity from other 
plant sections in 
accordance with the 
plant-specific design 
principles. During the 
test, an individual plant 
section is entirely or 
partially prevented 
from operating. The 
test demonstrates the 
capability of the other 
plant sections to meet 
the reactive power 
capacity requirement 
or implement active 
power curtailment if the 
reactive power 
capacity is insufficient. 
levels in accordance 
with VJV/SJV.  
9 Rapid down-regulation 
of active power  
Yes No 
(Does not apply to grid 
energy storage 
systems. See 
SJV2019.) 
10 Shut-down and start-up Yes.  
In addition to normal 
start-up and shut-
down, the test includes 
restoration to 
production following a 
10-minute network 
outage in which the 
power plant’s external 
electricity and 
telecommunications 
network connections 
are lost.  
Yes. 
Testing of the normal 
start-up and shut-
down of the plant 
section. 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 16 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
11 Fault ride-through The necessity of the test is considered on a 
case-by-case basis. 
 
Commissioning tests for type B hybrid power plants are conducted in the scope described 
in VJV2018 section 19.2. 
4.2  Monitoring period 
A monitoring period of at least 30 days must be arranged to demonstrate the continuous 
operation of the hybrid power plant’s central controller. A report must be prepared on the 
monitoring period to show that the plant’s controls function according to the principles in 
the plant’s system description.  
During the monitoring period, the power plant’s phase currents and voltages are 
measured and reported on the high-voltage side of each main transformer. The 
measurements are used to calculate the active and reactive power and frequency. The 
sampling rate of the measuring instruments must be at least 1 kHz, and the recording 
frequency must be at least 50 Hz. The power plant’s disturbance/oscillation recorders 
may be used for monitoring if they have suitable continuous measurement features. 
The largest network disturbance/incident during the monitoring period is selected and 
used to validate the simulation models. The voltage and frequency recording of the event 
from the connection point is repeated in the simulation model, and the responses of the 
power plant’s various plant sections are compared with the measurements in the 
corresponding situation. A representative network incident is agreed upon with Fingrid at 
the end of the monitoring period, and the validation results are included in the report. 
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 17 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
5  References 
/1/ Grid Code Specifications for Power Generating Facilities VJV2018, 
https://https://www.fingrid.fi/globalassets/dokumentit/en/customers/grid-
connection/grid-code-specifications-for-power-generating-facilities-vjv2018-.pdf 
(Accessed 17 October 2023) 
/2/ Grid Code Specifications for Grid Energy Storage Systems SJV2019, 
https://www.fingrid.fi/globalassets/dokumentit/en/customers/grid-connection/grid-
energy-storage-systems-sjv2019.pdf  (Accessed 17 October 2023) 
/3/ Reactive power requirements for power park modules and switched reactive 
power compensation, 
https://www.fingrid.fi/globalassets/dokumentit/fi/palvelut/kulutuksen-ja-tuotannon-
liittaminen-kantaverkkoon/voimalaitosten-loistehokapasiteettivaatimus-ja-
lisakompensointi.pdf (Accessed 17 October 2023) 
/4/ Supply of reactive power and maintenance of reactive power reserves, 
https://www.fingrid.fi/globalassets/dokumentit/en/customers/power-
transmission/supply-of-reactive-power-and-maintenance-of-reactive-power-
reserves-2021-id-269130.pdf (Accessed 17 October 2023) 
/5/ Modelling instruction for PSS/E and PSCAD models, 
https://www.fingrid.fi/globalassets/dokumentit/fi/palvelut/kulutuksen-ja-tuotannon-
liittaminen-kantaverkkoon/fingrid-modelling-instruction-for-psse-and-pscad-
models-2024_01_12-002.pdf (Accessed 17 October 2023) 
/6/ Real-time information exchange, 
https://www.fingrid.fi/globalassets/dokumentit/en/customers/power-
transmission/real-time-information-exchange_.pdf (Accessed 17 October 2023) 
 
 
 
 
 
 
Appendices  
 
 Appendix 1 Application examples 
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 18 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
Appendix 1 Application examples of hybrid power plants 
1  Example 1 
Hybrid plant: Wind farm with DFIG turbines PmaxWPP = 100 MW, solar power plant 
PmaxPV = 50 MW, and grid energy storage PmaxESS = 20 MW. The connection 
agreement states that the hybrid power plant’s rated capacity (Pmax) at the connection 
point is 130 MW. The power plant has an 8 km connecting line to the connection point 
specified in the connection agreement.  
 
  
Fingrid substation 
Connection point 
Connecting line Substation owned by the connecting party 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 19 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
• Control implementation 
▪ The hybrid power plant in this example has a plant-level central controller 
(“Hybrid PPC”) that measures and controls the plant’s active and reactive 
power and voltage at the high-voltage side of the high-voltage transformer. 
The active power is limited to 130 MW in all operating states. Reactive 
power slopes are implemented for the central controller.  
▪ Every plant section has a dedicated controller, which receives active and 
reactive power commands from the central controller. The controllers of 
the plant sections independently control their individual production units 
based on the terminal quantities of each production unit’s power converter.  
▪ The central controller is at least one order of magnitude (1/10) slower than 
controlling individual power converters to avoid cross-control. 
▪ The tap-changer in the power plant’s main transformer controls itself 
based on the 33 kV busbar voltage.   
• The hybrid power plant is a type D plant. Its reactive power capacity requirement 
is +/-0.33 x Pmax at the high-voltage terminals of the power plant’s main 
transformer (taking into account the voltage limits in Figures 1 and 2). In other 
words, it is +/- 0.33 x 130 MW → +/- 42.9 Mvar. This reactive power capacity is 
implemented as a combination of the reactive power capacities of various plant 
sections. The power converters are dimensioned accordingly. The plant does not 
have any mechanically switchable capacitors. 
• The reactive power capacity requirement applies when the largest plant section – 
in this case, the DFIG wind farm – operates above its minimum output of 5%, 
which is 5 MW. However, the reactive power capacities of the power converters 
belonging to the solar power plant and grid energy storage system are not 
restricted by software, even when the wind farm does not operate above its 
minimum output.  
• The solar power plant and grid energy storage system are capable of operating 
independently. In such a case, they operate under voltage control, adjusting to a 
voltage of 110 kV, and the reactive power capacity required for the main 
transformer’s high-voltage terminals is +/- 0.33 x 50 MW → +/- 16.7 Mvar for the 
solar power plant and +/- 0.33 x 20 MW → +/- 6.7 Mvar for the grid energy storage 
system. 
• The wind farm’s power converters are dimensioned so that when it operates at full 
power, the reactive power capacity of the solar power plant’s full power converters 
is used to produce the required reactive power capacity of +/- 0.33 x 100 MW → 
+/- 33.3 Mvar. 
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 20 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
2  Example 2 
Hybrid plant: A 30 MW wind farm and 15 MW grid energy storage system on the same 
transmission line connection to a 110 kV Fingrid transmission line. One main transformer 
of 31.5 MVA. The connection agreement states that the hybrid power plant’s rated 
capacity (Pmax) to the connection point is 30 MW. No connecting line, as the connecting 
party’s substation is located in the immediate vicinity of Fingrid’s transmission line.  
 
• Control implementation principle  
▪ The hybrid power plant in this example has a wind power plant park 
controller (“WPP PPC”) that operates as the plant-level central controller, 
measuring and controlling the plant’s active and reactive power and 
voltage at the high-voltage side of the high-voltage transformer. The active 
power is restricted to 30 MW in all operating states. Reactive power slopes 
are implemented for the central controller.  
▪ The grid energy storage system has a dedicated controller, which receives 
active and reactive power commands from the wind farm’s park controller. 
The controllers of the plant sections independently control their individual 
production units based on the terminal quantities of each production unit’s 
power converter.  
▪ The central controller is at least one order of magnitude (1/10) slower than 
controlling individual power converters to avoid cross-control. 
Connection point to Fingrid’s transmission line 
Fingrid’s substation A Fingrid’s substation A 
Substation owned by 
the connecting party 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 21 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
▪ The tap-changer in the power plant’s main transformer controls itself 
based on the 33 kV busbar voltage.   
• The hybrid power plant is a type D plant. Its reactive power capacity requirement 
is +/-0.33 x Pmax at the high-voltage terminals of the power plant’s main 
transformer (taking into account the voltage limits in Figures 1 and 2). In other 
words, it is +/- 0.33 x 30 MW → +/- 9.9 Mvar.  
 
• The reactive power capacity requirement applies when the largest plant section – 
in this case, the wind farm – operates above its minimum output. The turbines 
have full power converters (FCs), so the minimum output is 0 MW (0%). The 
reactive power capacity (voltage control) must be available when the power 
converters are in production-ready mode and connected to the grid.   
  
• The wind farm and grid energy storage system are capable of operating 
independently. In such a case, they operate under voltage control, adjusting to a 
voltage of 110 kV, and the reactive power capacity required for the main 
transformer’s high-voltage terminals is +/- 0.33 x 30 MW → +/- 9.9 Mvar for the 
wind power plant and +/- 0.33 x 15 MW → +/- 5.0 Mvar for the grid energy storage 
system. Capacitors are connected to the 33 kV busbar to allow the wind farm to 
operate independently. The capacitors supplement the power plant’s reactive 
power capacitor at high active powers. 
 
• The requirements for the fault current injection of power plants and grid energy 
storage systems are parameterised according to VJV2018 section 10.3.3 and 
SJV2019 section 10.3.3. As this plant has a transmission line connection, the 
hybrid power plant’s fault current injection must be limited according to YLE2021 
section 2.5 to 1.2 times the nominal current (300 ms from the onset of the fault). 
Therefore, when both plant sections are operating, the fault current must be 
limited to 1.2 x (30 MVA / sqrt(3) / 110 kV) = 189 A.     
  
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 22 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
3  Example 3 
Hybrid plant: 2 x 1.2 MW grid energy storage systems are added to an existing 2 x 
20 MW hydropower plant. Two 25 MVA generator transformers. The connection 
agreement states that the hybrid power plant’s rated capacity (Pmax) to the connection 
point is 40 MW. The power plant has a 110 kV Fingrid switchgear connection. No 
connecting line. The purpose of the grid energy storage system is to reduce the 
mechanical control movement of the water turbine when the hydropower plant operates in 
the reserve market. It does not increase the power plant’s rated capacity. 
 
• Control implementation principle  
▪ The hydropower plant’s 20 MW units (1 and 2) are independent units with 
dedicated water routes, so they are treated as separate power plants. 
Therefore, the hybrid power plant consists of a combination of one 
generator and a battery.     
▪ A voltage controller controls the hydropower plant’s generator and its 
excitation system, while a turbine controller controls the turbine. 
Connection point 
Fingrid’s substation 
Substation owned by the connecting party 
 
  Instruction / Application of the grid code 
specifications to hybrid power plants 
UNOFFICIAL 
TRANSLATION 17.10.2023 
  
   
  17 October 2023 23 (23) 
  
Fingrid Oyj 
Street address Postal address Telephone Fax Business ID 1072894-3, VAT reg. 
Läkkisepäntie 21 P.O. Box 530   firstname.lastname@fingrid.fi 
00620 Helsinki 00101 Helsinki, Finland +358 (0)30 3955000 +358 (0)30 3955196 www.fingrid.fi 
 
▪ The hybrid PPC is responsible for the frequency control offered to the 
reserve market. The hybrid PPC’s active power command to the grid 
energy storage system is coordinated with the hydropower generator’s 
turbine controller. A hybrid controller limits the power plant’s active power 
to the permitted level of 20 MW.    
• The hybrid power plant is a type D plant. In line with the requirements for 
synchronous machine power plants (VJV2018 section 12.2.2), its reactive power 
capacity requirement is +/-0.33 x Pmax at the high-voltage terminals of the power 
plant’s main transformer (taking into account the voltage limits in Figures 1 and 2). 
In other words, it is +/- 0.33 x 20 MW → +/- 6.7 Mvar.  
• The generator is responsible for meeting the reactive power capacity and 
operates under continuous voltage control.  
• The grid energy storage system is incapable of independent operation. It only 
operates when the hydropower plant supplies power. The grid energy storage 
system operates under power factor control (with the alignment cos φ ≈ 1) and 
does not contribute to voltage control. 
• The grid energy storage system’s fault current injection is parameterised 
according to SJV2019 section 10.3.3, and the grid energy storage system meets 
the fault ride-through requirement for a type D plant according to SJV2019 section 
10.5.2.     
    
 
  
 """,
    ),
    (
        "de_eeg_8a_flexible_netzanschluss",
        "EEG 2023 (Erneuerbare-Energien-Gesetz), § 8a -- Flexible Netzanschlussvereinbarungen",
        "https://www.gesetze-im-internet.de/eeg_2014/__8a.html",
        "DE",
        "de",
        # See in-document quality note: Absatz 1 is a verbatim quote,
        # Absaetze 2-3 are a faithful structural summary of the actual
        # fetched primary-source page, not a paraphrase from memory.
        """\
EEG 2023, § 8a -- Flexible Netzanschlussvereinbarungen
Quelle: gesetze-im-internet.de (offizielles Gesetzesportal der Bundesrepublik Deutschland)
Fundstelle: https://www.gesetze-im-internet.de/eeg_2014/__8a.html

HINWEIS ZUR INHALTSQUALITAET: Direkt von gesetze-im-internet.de abgerufen; Absatz 1
ist wortgetreu zitiert, Absaetze 2-3 sind eine inhaltsgetreue strukturierte
Zusammenfassung des tatsaechlichen Gesetzestexts (das Abruf-Tool liefert bei komplexen
Gliederungen eine Struktur-Zusammenfassung statt Rohtext) -- keine Paraphrase aus dem
Gedaechtnis, jede Aussage stammt direkt aus der abgerufenen Seite.

Absatz 1 (wortgetreu zitiert):
"Der Netzbetreiber und der Anlagenbetreiber koennen eine anschlussseitige Begrenzung
der maximalen Wirkleistungseinspeisung in das Netz vereinbaren." Der Anlagenbetreiber
muss die Einhaltung der vereinbarten Begrenzung durch technische Massnahmen
sicherstellen. Leistungsbegrenzungen koennen zeitfensterabhaengig sein und in ihrer
Hoehe variieren.

Absatz 2 (Struktur-Zusammenfassung): Die Vereinbarung muss sechs Kernbereiche regeln:
1. Hoehe der Leistungsbegrenzung(en)
2. Zeitfenster mit unterschiedlichen Begrenzungen
3. Dauer der Beschraenkungen
4. Technische Anforderungen zur Umsetzung
5. Haftung des Anlagenbetreibers bei Ueberschreitung der vereinbarten Grenze
6. "zum Einverstaendnis anderer Anlagenbetreiber oder Betreiber von Stromspeichern"
   (Zustimmung anderer Anlagen- bzw. Speicherbetreiber) -- bei gemeinsamem
   Netzanschlusspunkt sind zusaetzlich gemeinsame Verantwortlichkeits- und
   Haftungsregelungen zwischen mehreren Betreibern (z.B. Erzeugungsanlage +
   Batteriespeicher am selben Anschlusspunkt) erforderlich.

Absatz 3 (Struktur-Zusammenfassung): Der Netzbetreiber muss pruefen, ob eine flexible
Vereinbarung an einem alternativen Netzverknuepfungspunkt technisch und wirtschaftlich
guenstiger waere, und dem Anlagenbetreiber das Pruefergebnis mitteilen.

RELEVANZ FUER HYBRIDPROJEKTE: Absatz 2 Nr. 6 ist die einschlaegige Klausel fuer
BESS-Projekte, die sich einen Netzanschlusspunkt mit einer Wind- oder
Solaranlage teilen -- sie schreibt explizit eine Zustimmungs- und
gemeinsame-Haftungsregelung zwischen den Betreibern der verschiedenen Anlagentypen
am selben Anschlusspunkt vor ("grid overbuild" / gemeinsame Netzanschlusskapazitaet).""",
    ),
    (
        "de_baugb_35_batteriespeicher_privilegierung",
        "BauGB (Baugesetzbuch), § 35 Absatz 1 Nummer 11 -- Batteriespeicher-Privilegierung im Aussenbereich",
        "https://www.gesetze-im-internet.de/bbaug/__35.html",
        "DE",
        "de",
        # Nummer 11 (the operative clause) is a verbatim quote from
        # gesetze-im-internet.de; Absatz 1's introductory structure is a
        # faithful summary of the actual fetched page.
        """\
BauGB (Baugesetzbuch) § 35 Absatz 1 Nummer 11 -- Privilegierte Vorhaben im Aussenbereich
Quelle: gesetze-im-internet.de (offizielles Gesetzesportal der Bundesrepublik Deutschland)
Fundstelle: https://www.gesetze-im-internet.de/bbaug/__35.html

HINWEIS ZUR INHALTSQUALITAET: Nummer 11 ist wortgetreu zitiert (direkt von
gesetze-im-internet.de abgerufen); die einleitende Regelungsstruktur von Absatz 1
(dass eines der aufgezaehlten Privilegierungsmerkmale vorliegen muss, DAMIT ein
Vorhaben im Aussenbereich zulaessig ist) ist eine inhaltsgetreue Zusammenfassung.

Regelungsstruktur Absatz 1 (Zusammenfassung): Ein Vorhaben ist im Aussenbereich nur
zulaessig, wenn oeffentliche Belange nicht entgegenstehen, die ausreichende
Erschliessung gesichert ist, UND das Vorhaben unter eine der in Absatz 1 Nummer 1
bis Nummer 11 aufgezaehlten privilegierten Vorhabenarten faellt (Land-/
Forstwirtschaft, oeffentliche Versorgung, Wind-/Wasserkraft- und
Solarenergieanlagen unter bestimmten Bedingungen, kerntechnische Anlagen, usw.).

Nummer 11 (wortgetreu zitiert):
"der Speicherung von elektrischer Energie in einer Batteriespeicheranlage dient
und das Vorhaben in einem raeumlich-funktionalen Zusammenhang mit einer
vorhandenen Anlage zur Nutzung erneuerbarer Energien steht"

Auf Deutsch: Ein Vorhaben ist privilegiert im Aussenbereich zulaessig, wenn es der
Speicherung elektrischer Energie in einer Batteriespeicheranlage dient UND es in
einem raeumlich-funktionalen Zusammenhang mit einer bereits vorhandenen Anlage zur
Nutzung erneuerbarer Energien (z.B. Windenergie- oder Solaranlage) steht.

RELEVANZ FUER HYBRIDPROJEKTE: Dies ist die zentrale bau-/planungsrechtliche
Vorschrift, die einen Batteriespeicher direkt am Standort einer bestehenden
Wind- oder Solaranlage planungsrechtlich privilegiert -- ein echtes
Dual-Land-Use-Genehmigungsverfahren fuer co-located BESS+Erneuerbare-Projekte,
nicht nur eine allgemeine Regel fuer Speicher- und Erzeugungsanlagen getrennt.""",
    ),
    (
        "da_energinet_samplacerede_overplantede_krav",
        "Energinet, Tekniske krav til samplacerede og/eller overplantede elproducerende og -forbrugende anlaeg samt energilageranlaeg (Dok. 23/13192-1), godkendt af Forsyningstilsynet oktober 2025",
        "https://energinet.dk/media/wb1fexvz/tekniske-krav-til-samplacerede-og-eller-overplantede-elproducerende-og-forbrugende-anlaeg-samt-energilageranlaeg.pdf",
        "DA",
        "da",
        # Retrieved via direct PDF fetch + pypdf text extraction (same
        # extraction-tool limitation as the FI source above). Full 39-page
        # document, real primary source. Defines "samplaceret anlaeg"
        # (co-located facility) and "overplantet anlaeg" (overplanted
        # facility) -- production, storage, and consumption at one
        # connection point (PoC) -- and sets the technical requirements
        # (RfG/NC DC-referenced) for that shared connection point.
        """\
1/39 
 
Dok. 23/13192-1 Offentlig/Public 
Energinet 
Tonne Kjærsvej 65 
DK-7000 Fredericia 
 
+45 70 10 22 44 
info@energinet.dk  
CVR-nr. 39 31 49 59 
Dato: 
13. marts 2024 
 
Forfatter:  
KAB/CSH/MLG 
 
 
 
 
 
 
 
NOTAT 
TEKNISKE KRAV TIL SAMPLACEREDE OG/ELLER 
OVERPLANTEDE ELPRODUCERENDE OG  
-FORBRUGENDE ANLÆG SAMT ENERGILAGERANLÆG 
 
 
 
 
 
  
Dokumenttitel Tekniske krav til samplacerede og/eller overplantede elproduce-
rende og -forbrugende anlæg samt energilageranlæg  
Dokumentnummer 23/13192-1 
Målgruppe Det kollektive elsystems aktører 
Version Dokument- 
status 
Ejer Reviewer Godkender 
Navn Dato Navn Dato Navn Dato 
A Udkast CSH, KAB, 
MLG 06/02-2024 
CFJ, CVL, 
CXO, FBN, 
JHK, JST, 
JRH, MNR, 
SBS, SUD 
20/02-2024   
0 Endelig CSH, KAB 06/03-2024 FBN 06/03-2024 CFJ, JBO 11/03-
2024 

2/39 
 
Dok.23/13192-1 Offentlig/Public 
Indh old 
Indhold ................................................................................................... 2 
1. Læsevejledning ................................................................................ 4 
2. Nomenklatur ................................................................................... 5 
3. Introduktion og formål .................................................................... 6 
4. Termer og definitioner .................................................................... 7 
4.1 PoC ........................................................................................................................ 7 
4.2 Systembrugeren .................................................................................................... 7 
4.3 Samplacering ........................................................................................................ 7 
4.4 Anlægs- og udvekslingskapacitet .......................................................................... 7 
4.5 Overplanting ......................................................................................................... 8 
4.6 Samlede anlæg ...................................................................................................... 8 
4.7 Individuelle anlæg ................................................................................................. 8 
4.8 Selvstændige anlæg .............................................................................................. 8 
4.9 Forbrugstilstand .................................................................................................... 8 
4.10 Produktionstilstand ............................................................................................... 8 
5. Tekniske krav ................................................................................. 10 
5.1 Grænser for overskridelse af udvekslingskapaciteten ........................................ 10 
5.2 Tilslutningsproces ............................................................................................... 12 
5.3 Systemværn ........................................................................................................ 12 
5.4 Driftsspændingsområde ..................................................................................... 14 
5.5 Fault Ride Through (FRT) .................................................................................... 14 
5.6 Over Voltage Fault Ride Through (OVFRT) .......................................................... 15 
5.7 Reaktiv tillægsstrøm ........................................................................................... 16 
5.8 Post Fault Active Power Recovery (PFAPR) ......................................................... 17 
5.9 Rate of Change of Frequency (ROCOF) ............................................................... 19 
5.10 Limited frequency sensitivity mode – Overfrequency (LFSM-O) ........................ 19 
5.11 Limited frequency sensitivity mode – Underfrequency (LFSM-U) ...................... 20 
5.12 Low Frequency Demand Disconnection (LFDD) .................................................. 21 
5.13 Manuel aflastning ............................................................................................... 21 
5.14 Begrænsning af spændingsvariationer ved spændingssætning  ......................... 22 
5.15 Power Oscillation Damping (POD) ...................................................................... 23 
5.16 Elkvalitet ............................................................................................................. 24 
5.17 Aktiv effekt-referencepunkt ............................................................................... 25 
5.18 Aktiv effekt-reguleringsrampe ............................................................................ 26 
5.19 Reaktiv effekt-egenskaber .................................................................................. 27 
5.20 Reaktiveffektregulering ...................................................................................... 29 
5.21 Simuleringsmodel ............................................................................................... 30 
5.22 PMU-måling ........................................................................................................ 32 
5.23 Registrering af fejlhændelser (Transient Fault Recorder, TFR) ........................... 33 
5.24 Produktions/Forbrugstelegraf ............................................................................ 34 
5.25 Signalliste ............................................................................................................ 34 
5.26 Køreplaner og tilsvarende målinger.................................................................... 34 
5.27 Gensynkronisering .............................................................................................. 34 
6. Bilag ............................................................................................... 36 
6.1 Uddybelse af systemværnskrav .......................................................................... 36 
3/39 
 
Dok.23/13192-1 Offentlig/Public 
6.1.1 Præcisering for produktions- og energilageranlæg angående 
forbrugstilstand ...................................................................................... 36 
6.1.2 Præcisering for forbrugsanlæg angående produktionstilstand .............. 36 
6.2 Uddybelse af PFAPR-krav .................................................................................... 37 
6.2.1 Præcisering for produktions- og energilageranlæg angående 
forbrugstilstand ...................................................................................... 37 
6.2.2 Præcisering for forbrugsanlæg angående produktionstilstand .............. 37 
  
4/39 
 
Dok.23/13192-1 Offentlig/Public 
1. Læsevejledning 
Følgende dokument indeholder Energinets første udkast til justeringer af tilslutningskravene 
for transmissionstilsluttede produktions-, forbrugs- og energilageranlæg gældende for over-
plantede og/eller samplacerede anlæg. Dokumentet refererer gennemgående til eksisterende 
krav i eksisterende netregler. Det er udelukkende tilføjelser og ændringer af netregler rele-
vante for overplantede og/eller samplacerede anlæg, som beskrives eksplicit i dette dokument. 
Læsning og forståelse forudsætter derfor kendskab til kravene beskrevet i følgende af  
Energinets netregler. Dette dokument refererer gennemgående til de relevante afsnit i de eksi-
sterende netregler.  
 
• Krav jf. RfG – Bilag 1 Anmeldt til Forsyningstilsynet 21-12-2022 (Rev. 2C),  
dok. nr. 16/05118-120. 
• Bilag 1B Requirements for Generators (RfG) – Krav til simuleringsmodel, Rev 3 (Anmel-
delsesdokument), dok. nr. 16/05118-114. 
• Teknisk forskrift 3.2.7 – Krav til spændingskvalitet, spændingssætning og kobling for 
produktionsenheder i transmissionssystemet – Rev. 3, dok. nr. 18/03206-32. 
• DCC bilag 1 – Generelle tekniske krav for nettilslutning af forbrugs- og distributionssy-
stemer, Rev. 2C (Anmeldelsesdokument), dok. nr. 17/07437-82. 
• DCC Bilag 1.E – Krav for elkvalitet for transmissionstilsluttede distributionssystemer og 
forbrugsanlæg, Rev. 1B (Anmeldelsesdokument), dok. nr. 17/07437-83. 
• Teknisk forskrift 3.4.2 – Manuel aflastning af transmissionstilsluttede forbrugsanlæg, 
dok. nr. 20/05945-11. 
• Teknisk Forskrift 3.4.3 – Krav til Transmissionstilsluttede forbrugsanlæg, dok. nr. 
21/07383-74. 
• Teknisk Forskrift 3.3.1 – Krav til Energilageranlæg, Rev. 5, dok. nr. 24/01784-1. 
 
Der gøres opmærksom på, at dokumenterne angivet med ”anmeldelsesdokument” eller ”an-
meldt” er anmeldte opdateringer af netreglerne til Forsyningstilsynet. Disse træder endeligt i 
kraft, når Forsyningstilsynets godkendelse udstedes.  
 
Alle ovenstående dokumenter er tilgængelige på Energinets hjemmeside:  
https://energinet.dk/regler/el/nettilslutning/
  
  
5/39 
 
Dok.23/13192-1 Offentlig/Public 
2. Nomenklatur 
NC DC Network Code on Demand Connection 
FRT Fault Ride Through 
FSM Frequency Sensitivity Mode 
LFDD Low Frequency Demand Disconnection 
LFSM-O Limited Frequency Sensitivity Mode – Overfrequency 
LFSM-U Limited Frequency Sensitivity Mode – Underfrequency 
NTA Nettilslutningsaftale 
OVFRT Over Voltage Fault Ride Through 
PFAPR Post Fault Active Power Recovery 
PnD1 Summen af fysisk installeret forbrugskapacitet, anlægskapacitet   
PnD2 Forbrugsanlæggets maksimale effektniveau, ved hvilken compliance kan opnås. 
PnD3 Aktiv effekt, som må optages fra det kollektive elforsyningssystem aftalt i NTA.  
PnG1 Summen af fysisk installeret produktionskapacitet, anlægskapacitet.  
PnG2 Produktionsanlæggets maksimale effektniveau, ved hvilken compliance kan op-
nås. 
PnG3 Aktiv effekt, som må leveres til det kollektive elforsyningssystem aftalt i NTA. 
PoC Point of Connection, tilslutningspunkt 
POD Power Oscillation Damping 
NC RfG Network Code On Requirements For Grid Connection Of Generators 
ROCOF Rate Of Change Of Frequency 
TF Teknisk Forskrift 
TFR Transient Fault Recorder, også kaldet fejlskriver 
 
  
6/39 
 
Dok.23/13192-1 Offentlig/Public 
3. Introduktion og formål 
Ved ikrafttrædelsen af direkte-linje-bekendtgørelsen (bekendtgørelse nr. 437 af 27. april 2023 
med senere ændringer) i 2023 er det i et større omfang blevet muligt at tilslutte anlæg til det 
kollektive elsystem, som er etableret som samplacering af produktions-, forbrugs- og energila-
geranlæg. Som følge af forskellige tiltag er det desuden realiseret at kunne etablere et trans-
missionstilsluttet anlæg med et væsentligt niveau af overplanting. Overplanting og samplace-
ring er ligeledes blevet indtænkt i det statslige udbud Mere Havvind 2030 og koncessionsud-
buddet for Energiø Bornholm, hvori havvindskoncessionerne indeholder havvindsarealer, som 
muliggør overplanting. Denne nye type af tilslutning er ikke behandlet i de eksisterende god-
kendte netregler for hverken produktions-, forbrugs- eller energilageranlæg. Dette dokument 
beskriver krav, som Energinet vil anvende for overplantede og/eller samplacerede anlæg, der 
tilsluttes det kollektive elsystem, enten som del af Mere Havvind 2030, det isolerede AC-
system etableret på energiøer eller som øvrig transmissionstilslutning (særlige krav gælder for 
Energiøer og de anlæg, som tilsluttes det isolerede AC-system). De endeligt gældende krav vil 
blive implementeret i relevante tekniske forskrifter (TF’er) samt de nationalt godkendte krav i 
hhv. forordningerne Network Code on Requirements for grid connection for Generators (NC 
RfG) og Network Code on Demand Connection (NC DC) efter de påkrævede hørings- og god-
kendelsesprocesser herfor. Intentionen bag kravene er at tildele anlægsudviklerne størst mulig 
designfrihed under forudsætning af, at de krævede anlægsegenskaber og karakteristika leveres 
i PoC, så det samlede anlæg kan integreres og drives sikkert i det kollektive elsystem  uden øko-
nomiske konsekvenser herfor, samt at dansk og EU-lovgivning overholdes.  
 
Til at beskrive den nye type af anlæg, som overplanting og samplacering skaber rammerne for, 
har det vist sig nødvendigt at introducere termer og definitioner, som ikke tidligere har været 
del af Energinets netregler. De anvendte termer og definitioner er defineret i Afsnit 4. En af de 
væsentligste termer er det samlede anlæg. Det samlede anlæg består af et antal individuelle 
produktions-, forbrugs- og/eller energilageranlæg, og har en aftalt udvekslingskapacitet (i MW) 
i PoC, som er mindre end eller lig med den samlede anlægskapacitet. Kravene for overplanting 
og samplacering har til hensigt at sikre, at det samlede anlæg overordnet set har de samme 
tekniske egenskaber som et selvstændigt anlæg med en anlægskapacitet lig med det samlede 
anlægs udvekslingskapacitet. Det betyder bl.a., at krav til produktions-, forbrugs- og energila-
geranlæg forenes, så det samlede anlægs opførsel er et koordineret respons, på trods af at det 
består af forskellige anlægstyper.  
 
De tekniske krav til overplanting og samplacering beskrives i Afsnit 5. Beskrivelsen af hvert krav 
indeholder en overordnet hensigt for de påkrævede tekniske egenskaber af det samlede an-
læg, samt udmøntning af krav til hver af anlægstyperne: produktionsanlæg, forbrugsanlæg 
og/eller energilageranlæg, som kan indgå i det samlede anlæg. Det er kravene for hver anlægs-
type, som vil blive implementeret i de tilhørende netregler, herunder NC RfG, NC DC og TF’er. 
Denne udmøntning af krav på hver anlægstype i forskellige netregler er vurderet nødvendig for 
at overholde rammerne fastsat i EU-forordningerne NC RfG og NC DC. Ingen af forordningerne 
giver mulighed for at samle de tekniske krav til samplacerede anlæg i ét enkelt sæt af netreg-
ler, da forordningerne principielt regulerer anlæggene særskilt. Der kan eventuelt senere udar-
bejdes ikkebindende vejledningsmateriale, der beskriver samplaceringskravene i et samlet for-
mat.  
  
7/39 
 
Dok.23/13192-1 Offentlig/Public 
4. Termer og definitioner 
4.1 PoC 
”Tilslutningspunkt”, også kaldet ”Point of Connection” eller ”PoC”, er den grænseflade, hvor 
det samlede anlæg er tilsluttet det kollektive elsystem, hvor tilslutningskrav er gældende , og 
anlægs opførsel evalueres for compliance. 
 
4.2 Systembrugeren 
Energinet kræver i nettilslutningsaftalen, at der er én part, kaldet systembrugeren, som er an-
svarlig for det samlede anlæg, som er tilsluttet i installationen bag PoC. Systembrugeren er den 
fysiske eller juridiske person, som har de(t) fulde juridiske, fysiske, faglige og operationelle an-
svar, kompetencer og kontrolbeføjelser til at varetage driften og have ansvaret for det samlede 
anlæg, som er omfattet af nettilslutningsaftalen, og som er tilsluttet transmissionssystemet. 
 
4.3 Samplacering  
Samplacering realiseres, når produktionsanlæg, forbrugsanlæg og/eller energilageranlæg til-
sluttes i samme tilslutningspunkt (PoC) i systembrugerens installation og bag samme elmåler, 
så øjebliksafregning med en måleenhed er muligt.  
 
4.4 Anlægs- og udvekslingskapacitet 
For overplantede anlæg er det væsentligt at skelne mellem det samlede anlægs kapacitet for 
forbrug og/eller produktion af aktiv effekt, kaldet anlægskapacitet, og den aftalte maksimale 
udveksling af forbrug og/eller produktion med det kollektive elsystem, kaldet udvekslingskapa-
citet. Der skelnes mellem det samlede anlægs anlægskapacitet og udvekslingskapacitet for hhv. 
optag af aktiv effekt (forbrug) og levering af aktiv effekt (produktion) med nedenstående para-
metre. Det samlede anlægs anlægskapacitet for forbrug benævnes P
nD1. I tilfælde, hvor det 
samlede anlæg ikke kan opnå compliance i forhold til de tekniske krav ved effektniveauet PnD1, 
fastsættes det maksimale effektniveau, hvorved compliance kan opnås som PnD2. Der kan ved 
indgåelse af NTA aftales et maksimalt effektniveau for aktiv effekt optag fra det kollektive elsy-
stem benævnt PnD3, der er lavere end hhv. PnD1 og PnD2. Tilsvarende parametre for produktion 
er defineret og benævnt hhv. PnG1, PnG2 og PnG3.   
 
Anlægskapaciteten af et energilageranlæg, der samplaceres med produktions- og/eller for-
brugsanlæg, vil blive medregnet i parametrene PnD1 og PnG1, hvis energilageranlægget anvendes 
til f.eks. markedsprodukter. Hvis energilageranlæg agerer støtteudstyr for compliance med tek-
niske krav, inkluderes energilageranlæggets anlægskapacitet ikke i PnD1 og PnG1. 
 
 
 
PnD1 Summen af fysisk installeret forbrugskapacitet, anlægskapacitet.  
PnD2 Forbrugsanlæggets maksimale effektniveau, ved hvilken compliance kan opnås. 
PnD3 Aktiv effekt, som må optages fra det kollektive elforsyningssystem, aftalt i NTA. 
PnG1 Summen af fysisk installeret produktionskapacitet, anlægskapacitet.  
PnG2 Produktionsanlæggets maksimale effektniveau, ved hvilken compliance kan op-
nås. 
PnG3 Aktiv effekt, som må leveres til det kollektive elforsyningssystem, aftalt i NTA. 
 

8/39 
 
Dok.23/13192-1 Offentlig/Public 
4.5 Overplanting 
Overplanting realiseres, når forbrugs-, produktions- eller energilageranlæggets samlede instal-
lerede effekt (anlægskapacitet) er større end den aftalte udvekslingskapacitet, dvs.  PnG1 > PnG3 
og/eller PnD1 > PnD3. 
 
4.6 Samlede anlæg 
Det samlede anlæg består af et antal individuelle anlæg. Det samlede anlæg har en aftalt ud-
vekslingskapacitet med det kollektive elsystem, der er mindre end eller lig med den samlede 
anlægskapacitet.  
 
4.7 Individuelle anlæg 
Individuelle anlæg er enten produktions-, forbrugs- eller energilageranlæg, der indgår i det 
samlede anlæg som led i samplacering og overplanting. I det samlede anlæg vil der i regulato-
risk forstand kun være et enkelt produktionsanlæg, et enkelt forbrugsanlæg og/eller et enkelt 
energilageranlæg. Dette princip er illustreret på Figur 1. 
 
4.8 Selvstændige anlæg 
Selvstændige anlæg er energilager-, produktions- eller forbrugsanlæg, som er tilsluttet det kol-
lektive elsystem direkte uden samplacering.  
 
4.9 Forbrugstilstand 
Det samlede anlæg betragtes som værende i forbrugstilstand, når det samlede anlæg optager 
aktiv effekt fra det kollektive elsystem. 
 
4.10 Produktionstilstand 
Det samlede anlæg betragtes som værende i produktionstilstand, når det samlede anlæg leve-
rer aktiv effekt til det kollektive elsystem.  
 
 
9/39 
 
Dok.23/13192-1 Offentlig/Public 
 
Figur 1: Illustration af en ikkeudtømt sammensætning af anlægstyper i et samplaceret anlæg. 
  
  

10/39 
 
Dok.23/13192-1 Offentlig/Public 
5. Tekniske krav 
De følgende afsnit beskriver de enkeltstående tekniske krav, som vil blive implementeret for 
overplanting og samplacering. Alle krav er skrevet med henblik på udmøntning for hver af de 
tre anlægstyper: produktions-, forbrugs- og energilageranlæg, i de tilhørende netregler.  
Hvert afsnit er opbygget med følgende struktur: 
 
1. Overordnet beskrivelse af kravets hensigt 
2. Reference til eksisterende krav til selvstændige anlæg 
3. Udmøntning af overplantings- og samplaceringskrav til produktionsanlæg 
4. Udmøntning af overplantings- og samplaceringskrav til forbrugsanlæg 
5. Udmøntning af overplantings- og samplaceringskrav til energilageranlæg. 
 
Udmøntning af overplantings- og samplaceringskrav tager udgangspunkt i krav til selvstændige 
anlæg i eksisterende netregler. Heri er nogle af de påkrævede anlægsegenskaber specificeret 
på basis af anlæggets nominelle effekt, også kaldet Pn, hvilket svarer til PnG1 og PnD1 defineret 
for hhv. overplantede produktions- og forbrugsanlæg. Som eksempel er de påkrævede reaktiv 
effekt-egenskaber for et produktionsanlæg ved 1,00 pu driftsspænding at kunne levere Q = ± 
0,33·Pn. For overplantede anlæg vil sådan skalering af påkrævede anlægsegenskaber blive æn-
dret til en skalering på basis af enten PnD3 eller PnG3. Som eksempel vil et overplantet produkti-
onsanlæg med anlægskapacitet PnG1 = 200 MW og udvekslingskapacitet PnG3 = 100 MW blive 
påkrævet at have reaktiv effekt-egenskaber skaleret på basis af PnG3 frem for PnG1, så produkti-
onsanlægget skal kunne levere Q = ±0,33·100 MW = ±33 Mvar. 
 
Det er systembrugerens ansvar, at det samlede anlæg overholder de tekniske krav, og at det 
dokumenteres, at kravene overholdes. Energinet gennemgår dokumentation og træffer afgø-
relse om, hvorvidt overholdelse af tekniske krav er tilstrækkeligt opfyldt forud for tildeling af 
driftstilladelser. Energinet kan til enhver tid kræve opdateret verifikation og dokumentation 
for, at det samlede anlæg opfylder de gældende tekniske krav. 
 
5.1 Grænser for overskridelse af udvekslingskapaciteten  
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
fastsættes grænser for det samlede anlægs tilladte overskridelse af udvekslingskapaciteten i 
PoC. Disse fastsættes for at sikre, at anlæg med større anlægskapacitet end aftalt udvekslings-
kapacitet ikke kan overbelaste udstyr i det kollektive elsystem skadeligt, selv ved utilsigtede 
hændelser. Overskridelse af udvekslingskapacitet fastsættes som strømgrænser for at tage 
højde for driftssituationer, hvor spændinger er uden for normaldriftsområdet. Grænserne er 
defineret i Figur 2. Karakteristikken begynder (tiden 0,0 s) i det øjeblik, hvor det samlede anlæg 
udveksler mere end 1,1 pu strøm i PoC. Strømgrænserne er angivet i per unit (pu) på basis af 
I
nominel, defineret herunder.  
 
11/39 
 
Dok.23/13192-1 Offentlig/Public 
 
Figur 2: Strømgrænser for overskridelse af udvekslingskapacitet. Værdier er angivet i Tabel 1. 
 
Tidsinterval [s] Strøm i PoC [pu] 
0,0 – 0,1 2,00 
0,1 – 5,0 Lineært aftagende fra 1,50 til 1,25 
5,0 – 10,0 1,25 
10,0 – 30,0 1,10 
> 30,0 1,00 
Tabel 1: Definition af strømgrænser for overskridelser af udvekslingskapacitet .  
Karakteristikken begynder (tiden 0,0 s) i det øjeblik,  
hvor det samlede anlæg udveksler mere end 1,1 pu strøm i PoC. 
 
Grænser for overskridelse af udvekslingskapacitet angives i per unit med baseværdien Inominel, 
som er defineret ved følgende:  
 
 𝐼𝐼nominel = 𝑆𝑆PoC
√3 ⋅ 𝑈𝑈norm,min
  
Hvor:   
 𝑆𝑆PoC = �𝑃𝑃2 + 𝑄𝑄2  
 𝑃𝑃 = Største af PnG3 og PnD3  
 𝑄𝑄 = 0,33 ⋅ 𝑃𝑃𝑛𝑛𝑛𝑛3  
 𝑈𝑈𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛,𝑛𝑛 𝑚𝑚𝑛𝑛 = 𝑋𝑋 ⋅ 𝑈𝑈𝑛𝑛 
𝑋𝑋 = �
0,968 for 𝑈𝑈𝑛𝑛 mellem 110 − 300 kV i DK1
0,90 for 𝑈𝑈𝑛𝑛 mellem 110 − 300 kV i DK2
0,90 for 𝑈𝑈𝑛𝑛 mellem 300 − 400 kV
 
 
 
Eksisterende krav 
Ingen. 
 
Produktionsanlæg: 
Overplantede produktionsanlæg må ikke udveksle strøm med det kollektive elsystem, som 
overskrider strømgrænserne defineret i Figur 2.  
 
Produktionsanlæg, der er samplacerede med energilager- og/eller forbrugsanlæg, skal i koordi-
nation med de øvrige samplacerede anlæg sikre, at det samlede anlæg ikke overskrider 

12/39 
 
Dok.23/13192-1 Offentlig/Public 
strømgrænserne defineret i Figur 2. Det er systembrugerens ansvar, at der implementeres en 
teknisk løsning til koordinering af det samlede anlægs strøm udvekslet med det kollektive elsy-
stem. 
 
Grænser for overskridelse af udvekslingskapaciteten gælder i alle situationer. Kravet giver ikke 
undtagelse for øvrige krav til strømudveksling, f.eks. PFAPR eller reaktiv tillægsstrøm.  
 
Energilageranlæg: 
Overplantede energilageranlæg må ikke udveksle strøm med det kollektive elsystem, som 
overskrider strømgrænserne defineret i Figur 2.  
 
Energilageranlæg, der er samplacerede med produktions- og/eller forbrugsanlæg, skal i koordi-
nation med de øvrige samplacerede anlæg sikre, at det samlede anlæg ikke overskrider strøm-
grænserne defineret i Figur 2. Det er systembrugerens ansvar, at der implementeres en teknisk 
løsning til koordinering af det samlede anlægs strøm udvekslet med det kollektive elsystem. 
 
Grænser for overskridelse af udvekslingskapaciteten gælder i alle situationer. Kravet giver ikke 
undtagelse for øvrige krav til strømudveksling, f.eks. PFAPR eller reaktiv tillægsstrøm.  
 
Forbrugsanlæg: 
Overplantede forbrugsanlæg må ikke udveksle strøm med det kollektive elsystem, som over-
skrider strømgrænserne defineret i Figur 2.  
 
Forbrugsanlæg, der er samplacerede med energilager- og/eller produktionsanlæg, skal i koor-
dination med de øvrige samplacerede anlæg sikre, at det samlede anlæg ikke overskrider 
strømgrænserne defineret i Figur 2. Det er systembrugerens ansvar, at der implementeres en 
teknisk løsning til koordinering af det samlede anlægs strøm udvekslet med det kollektive elsy-
stem. 
 
Grænser for overskridelse af udvekslingskapaciteten gælder i alle situationer. Kravet giver ikke 
undtagelse for øvrige krav til strømudveksling, f.eks. PFAPR eller reaktiv tillægsstrøm.  
 
5.2 Tilslutningsproces  
Der tages afsæt i de forordningsbestemte tilslutningsprocesser. Tilslutningsprocessen anven-
des på det samlede anlæg.  
Behovet for anvendelse af successiv tilslutning vurderes og aftales med Energinet.  
 
5.3 Systemværn 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
skal de individuelle anlæg være udstyret med systemværn. Kravene har til formål at sikre, at 
det samlede anlæg giver et fælles systemværnsrespons, som understøtter det kollektive elsy-
stems stabilitet i særligt kritiske driftssituationer. Systemværnenes foruddefinerede regule-
ringstrin baseres på udvekslingskapaciteten (P
nG3 og PnD3). Ved aktivering af systemværn tages 
udgangspunkt i, om det samlede anlæg er i produktionstilstand eller forbrugstilstand. Et even-
tuelt systemværn aktiveres ved hjælp af ét signal til det samlede anlæg og ikke flere signaler til 
samplacerede individuelle anlæg. 
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 14.5.a.i & 15.6.d 
13/39 
 
Dok.23/13192-1 Offentlig/Public 
• Energilageranlæg: TF 3.3.1 § 58  
• Forbrugsanlæg: TF 3.4.3 § 5. 
 
Produktionsanlæg: 
Produktionsanlæg skal opfylde systemværnskravene jf. NC RfG art. 15.6.d angående foruddefi-
nerede reguleringstrin. For overplantede produktionsanlæg skaleres foruddefinerede regule-
ringstrin for systemværnet med udvekslingskapaciteten (PnG3) frem for installeret effekt (PnG1).  
 
Hvis produktionsanlæg samplaceres med energilager- og/eller forbrugsanlæg, skal produkti-
onsanlægget bidrage til, at det samlede anlæg opnår de foruddefineret reguleringstrin for ef-
fektudveksling i PoC i tilfælde af systemværnsaktivering. Systembrugeren har til ansvar at koor-
dinere bidrag fra de samplacerede anlæg, så det samlede anlæg følger kravene til  påbegyn-
delse af regulering, fuldendt regulering, reguleringstrin og nøjagtighed. Kravene differentieres 
afhængigt af, om det samlede anlæg var i produktions- eller forbrugstilstand forud for system-
værnsaktivering: 
• Produktionstilstand: NC RfG art. 15.6.d  
• Forbrugstilstand: TF 3.4.3 § 5. 
  
Kravene til forbrugstilstand er angivet i bilag i Afsnit 6.1.1, og disse tilføjes til NC RfG i forbin-
delse med endelig implementering af krav til overplanting og samplacering. 
 
Hvis der, i produktionsanlægget indgår elproducerende enheder med vind som primær energi, 
skal krav om ”automatisk nedreguleringsfunktion af aktiv effekt ved stopvindhastighed” følges. 
Dette krav skaleres i udgangspunktet med udvekslingskapaciteten (P
nG3) for overplantede 
og/eller samplacerede produktionsanlæg. For produktionsanlæg bestående af enheder med 
forskellige typer af primær energi, kan Energinet fastsætte alternativ skaleringsfaktor på basis 
af forholdet mellem installeret kapacitet af vindbaserede elproducerende enheder i forhold til  
den samlede kapacitet af installeret produktionskapacitet (P
nG1). 
 
Energilageranlæg: 
Energilageranlæg skal være udstyret med systemværn jf. TF 3.3.1 § 58. For overplantede ener-
gilageranlæg skaleres de foruddefinerede reguleringstrin for systemværnet med udvekslingska-
paciteten (PnG3 og PnD3) frem for installeret effekt (PnG1) og (PnD1). 
 
Hvis energilageranlæg samplaceres med produktions- og/eller forbrugsanlæg, skal energilager-
anlægget bidrage til, at det samlede anlæg opnår de foruddefinerede reguleringstrin for effekt-
udveksling i PoC i tilfælde af systemværnsaktivering. Systembrugeren har til ansvar at koordi-
nere bidrag fra de samplacerede anlæg, så det samlede anlæg følger kravene til påbegyndelse 
af regulering, fuldendt regulering, sætpunkt og nøjagtighed. Kravene differentieres afhængigt 
af, om det samlede anlæg var i produktions- eller forbrugstilstand forud for systemværnsakti-
vering: Bemærk, at kravene til systemværn er ens i NC RfG art. 15.6.d og TF 3.3.1 § 58. 
• Produktionstilstand: TF 3.3.1 § 58  
• Forbrugstilstand: TF 3.4.3 § 5. 
  
Kravene til forbrugstilstand er angivet i bilag i Afsnit 6.1.1, og disse tilføjes til TF 3.3.1 i forbin-
delse med endelig implementering af krav til overplanting og samplacering. 
 
Forbrugsanlæg: 
14/39 
 
Dok.23/13192-1 Offentlig/Public 
Forbrugsanlæg skal være udstyret med systemværn jf. TF 3.4.3 § 5. For overplantede forbrugs-
anlæg skaleres de foruddefinerede reguleringstrin for systemværnet med udvekslingskapacite-
ten (PnD3) frem for installeret effekt (PnD1). 
 
Hvis forbrugsanlægget samplaceres med energilager- og/eller produktionsanlæg, skal forbrugs-
anlægget bidrage til, at det samlede anlæg opnår de foruddefinerede reguleringstrin for effekt-
udveksling i PoC i tilfælde af systemværnsaktivering. Systembrugeren har til ansvar at koordi-
nere bidrag fra de samplacerede anlæg, så det samlede anlæg følger kravene til påbegyndelse 
af regulering, fuldendt regulering, reguleringstrin og nøjagtighed. Kravene differentieres af-
hængigt af, om det samlede anlæg var i produktions- eller forbrugstilstand forud for system-
værnsaktivering:  
• Produktionstilstand: NC RfG art. 15.6.d  
• Forbrugstilstand: TF 3.4.3 §5. 
  
Kravene til produktionstilstand er angivet i bilag i Afsnit 6.1.2, og disse tilføjes til TF 3.4.3 i for-
bindelse med endelig implementering af krav til overplanting og samplacering. 
 
5.4 Driftsspændingsområde  
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg ensrettes kravene til 
driftsspændingsområde. Herved sikres, at der ikke forekommer udkobling af dele af det sam-
lede anlæg, når det drives i normaldriftsområdet. 
 
Eksisterende krav  
• Produktionsanlæg: NC RfG art. 16.2.a 
• Energilageranlæg: TF 3.3.1 § 69  
• Forbrugsanlæg: NC DC art. 13.1. 
 
Produktionsanlæg: 
Ingen ændring. 
 
Energilageranlæg: 
Ingen ændring.  
 
Forbrugsanlæg: 
Hvis forbrugsanlægget samplaceres med energilager- og/eller produktionsanlæg i DK1 (det 
kontinentaleuropæiske synkronområde), pålægges forbrugsanlægget, at det skal forblive til-
sluttet til det kollektive elsystem i minimum 60 minutter, hvis spændingen i PoC er inden for 
intervallet 0,85 – 0,90 pu.  
 
5.5 Fault Ride Through (FRT) 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
pålægges de individuelle anlæg robusthedskrav. De påkrævede FRT-egenskaber har til formål 
at sikre, at det samlede anlæg ikke forårsager driftsforstyrrelser i det kollektive elsystem grun-
det uhensigtsmæssig udkobling af dele af det samlede anlæg ifm. et fejlforløb.  
   
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 16.3.a.i 
• Energilageranlæg: TF 3.3.1 § 128 og § 129 
• Forbrugsanlæg: TF 3.4.3 § 11. 
15/39 
 
Dok.23/13192-1 Offentlig/Public 
 
Produktionsanlæg: 
Produktionsanlægget skal overholde kravene i NC RfG art. 16.3.a.i for FRT-egenskaber i PoC.  
 
Hvis produktionsanlægget samplaceres med et energilager- og/eller et forbrugsanlæg, tillades 
det, at en delmængde af produktionsanlægget udkobler under forudsætning af, at alle øvrige 
tekniske krav under og efter hændelsen er overholdt, herunder at det samlede anlæg er i stand 
til at vende tilbage til samme driftspunkt for udveksling af aktiv effekt (i PoC) som før hændel-
sen, som udløste FRT. 
  
Energilageranlæg: 
Energilageranlægget skal overholde kravene i TF 3.3.1 § 128 og § 129 for FRT-egenskaber i 
PoC. 
 
Hvis energilageranlægget samplaceres med et produktions- og/eller et forbrugsanlæg, tillades 
det, at en delmængde af energilageranlægget udkobler under forudsætning af, at alle øvrige 
tekniske krav under og efter hændelsen er overholdt, herunder at det samlede anlæg er i stand 
til at vende tilbage til samme driftspunkt for udveksling af aktiv effekt (i PoC) som før hændel-
sen, som udløste FRT. 
 
Forbrugsanlæg: 
Forbrugsanlægget skal overholde kravene i TF 3.4.3 § 11 for FRT-egenskaber i PoC. 
 
Hvis forbrugsanlægget samplaceres med et produktions- og/eller et energilageranlæg, tillades 
det, at en delmængde af forbrugsanlægget udkobler under forudsætning af, at alle øvrige tek-
niske krav under og efter hændelsen er overholdt, herunder at det samlede anlæg er i stand til 
at vende tilbage til samme driftspunkt for udveksling af aktiv effekt (i PoC) som før hændelsen, 
som udløste FRT. 
 
5.6 Over Voltage Fault Ride Through (OVFRT) 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
pålægges de individuelle anlæg robusthedskrav. De påkrævede OVFRT-egenskaber har til for-
mål at sikre, at det samlede anlæg ikke forårsager driftsforstyrrelser i det kollektive elsystem 
grundet uhensigtsmæssig udkobling af dele af det samlede anlæg ifm. et overspændingsforløb.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 16.3.a.i (anmeldt) 
• Energilageranlæg: TF 3.3.1 § 72 
• Forbrugsanlæg: Ingen eksisterende krav. 
 
Produktionsanlæg: 
Produktionsanlæggets egenskab til OVFRT skal i PoC som minimum overholde kravene i NC RfG 
art. 16.3.a.i.  
 
Hvis produktionsanlægget samplaceres med et energilager- og/eller et forbrugsanlæg, tillades 
det, at en delmængde af produktionsanlægget udkobler under forudsætning af, at alle øvrige 
tekniske krav under og efter hændelsen er overholdt, herunder at det samlede anlæg er i stand 
til at vende tilbage til samme driftspunkt for udveksling af aktiv effekt (i PoC) som før hændel-
sen, som udløste OVFRT. 
 
16/39 
 
Dok.23/13192-1 Offentlig/Public 
Energilageranlæg: 
Energilageranlæggets egenskab til OVFRT skal i PoC som minimum overholde kravene i TF 3.3.1 
§ 73. 
 
Hvis energilageranlægget samplaceres med et produktions- og/eller et forbrugsanlæg, tillades 
det, at en delmængde af energilageranlægget udkobler under forudsætning af, at alle øvrige 
tekniske krav under og efter hændelsen er overholdt, herunder at det samlede anlæg er i stand 
til at vende tilbage til samme driftspunkt for udveksling af aktiv effekt (i PoC) som før hændel-
sen, som udløste OVFRT. 
 
Forbrugsanlæg: 
Forbrugsanlæg bliver pålagt samme OVFRT-karakteristik som produktionsanlæg (NC RfG art. 
16.3.a.i (anmeldt)). 
 
Hvis forbrugsanlægget samplaceres med et produktions- og/eller et energilageranlæg, tillades 
det, at en delmængde af forbrugsanlægget udkobler under forudsætning af, at alle øvrige tek-
niske krav under og efter hændelsen er overholdt, herunder at det samlede anlæg er i stand til 
at vende tilbage til samme driftspunkt for udveksling af aktiv effekt (i PoC) som før hændelsen, 
som udløste OVFRT. 
 
5.7 Reaktiv tillægsstrøm  
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg pålægges anlæggene 
krav til reaktiv tillægsstrøm i PoC som beskrevet nedenfor. Kravene har til formål at sikre, at 
det samlede anlæg yder et koordineret respons af levering af reaktiv tillægsstrøm for at under-
støtte det kollektive elsystem under et fejlforløb.  
   
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 20.2.b 
• Energilageranlæg: TF 3.3.1 §133 og §134 
• Forbrugsanlæg: Ingen eksisterende krav. 
  
Produktionsanlæg: 
Produktionsanlægget skal supplere reaktiv tillægsstrøm under fejlforløb jf. kravene i NC RfG 
art. 20.2.b. For overplantede produktionsanlæg ændres kravet således, at den påkrævede re-
aktive tillægsstrøm IQ/In og tolerance skaleres med InG3 beregnet på basis af udvekslingskapaci-
teten i PoC (PnG3) frem for In beregnet på basis af produktionsanlæggets installerede effekt 
(PnG1).  
  
Hvis produktionsanlægget samplaceres med energilager- og/eller forbrugsanlæg, tillades det, 
at de samplacerede anlæg bidrager til opfyldelse af kravet til reaktiv tillægsstrøm. Det er en 
forudsætning herfor, at reaktiv tillægsstrøm fra produktionsanlægget og de bidragende anlæg 
koordineres således, at den totale reaktive tillægsstrøm i PoC fra det samlede anlæg opfylder 
den krævede karakteristik jf. RfG art. 20.2.b. Kravet til reaktiv tillægsstrøm bortfalder, hvis det 
samlede anlæg kun har trækningsret, dvs. P
nG3 er lig 0. Det er systembrugerens ansvar, at der 
implementeres en teknisk løsning til koordinering af det samlede anlægs reaktive tillægsstrøm i 
PoC. 
  
Energilageranlæg: 
Energilageranlæg skal have reaktiv tillægsstrøm-egenskaber jf. kravene i TF 3.3.1 § 133 og § 
134 for hhv. DK1 og DK2. For overplantede energilageranlæg ændres kravet således, at den 
17/39 
 
Dok.23/13192-1 Offentlig/Public 
påkrævede reaktive tillægsstrøm IQ/In og tolerance skaleres med InG3 beregnet på basis af ud-
vekslingskapaciteten i PoC (PnG3) frem for In beregnet på basis af energilageranlæggets installe-
rede effekt (PnG1). 
  
Hvis energilageranlægget samplaceres med et forbrugsanlæg, tillades det, at det samplacerede 
forbrugsanlæg bidrager til opfyldelse af kravet til reaktiv tillægsstrøm. Det er en forudsætning 
herfor, at reaktiv tillægsstrøm fra energilageranlægget og det bidragende forbrugsanlæg koor-
dineres således, at den totale reaktive tillægsstrøm i PoC fra det samlede anlæg opfylder den 
krævede karakteristik jf. TF 3.3.1 § 133 og § 134 for hhv. DK1 og DK2. Kravet til reaktiv tillægs-
strøm bortfalder hvis det samlede anlæg kun har trækningsret, dvs. PnG3 er lig 0. Det er system-
brugerens ansvar, at der implementeres en teknisk løsning til koordinering af det samlede an-
lægs reaktive tillægsstrøm i PoC. 
  
Hvis energilageranlægget samplaceres med et produktionsanlæg, kræves det, at den reaktive 
tillægsstrøm fra energilageranlægget og produktionsanlægget koordineres således, at den to-
tale reaktive tillægsstrøm i PoC fra det samlede anlæg opfylder den krævede karakteristik fra 
kravene i NC RfG art. 20.2.b og TF 3.3.1 §133 og §134. Bemærk, at reaktiv tillægsstrøm-kravet 
er ens i NC RfG art. 20.2.b og TF 3.3.1 §133 og §134. Det er systembrugerens ansvar, at der im-
plementeres en teknisk løsning til koordinering af det samlede anlægs reaktive tillægsstrøm i 
PoC. 
 
Forbrugsanlæg: 
Forbrugsanlægget må ikke påvirke den udvekslede strøm under et hændelsesforløb, så det 
samlede anlægs respons afviger fra den påkrævede reaktive tillægsstrøm. 
 
Hvis systembrugeren vælger at udnytte forbrugsanlæggets egenskaber som bidrag til opfyl-
delse af energilageranlæggets og/eller produktionsanlæggets påkrævede reaktive tillægsstrøm-
egenskaber, er det systembrugerens ansvar, at der implementeres en teknisk løsning til koordi-
nering af det samlede anlægs reaktive tillægsstrøm i PoC. 
 
5.8 Post Fault Active Power Recovery (PFAPR) 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
skal de individuelle anlæg kunne udføre PFAPR. PFAPR-responset baseres på udvekslingskapa-
citeten (P
nG3 og PnD3). PFAPR-responset skal sikre, at det samlede anlæg efter et indsvingnings-
forløb opnår normal effektudveksling, når driftsforholdene i PoC er tilbage i området kontinu-
ert drift. Ved aktivering af PFAPR tages udgangspunkt i, om det samlede anlæg er i produkti-
ons- eller forbrugstilstand.  
 
Eksisterende krav 
 
• Produktionsanlæg: NC RfG art. 20.3.a (anmeldt) 
• Energilageranlæg: TF 3.3.1 §117 og §124 
• Forbrugsanlæg: TF 3.4.3 §12. 
 
Produktionsanlæg: 
Produktionsanlægget skal have PFAPR-egenskaber jf. kravene i NC RfG art. 20.3.a. For over-
plantede produktionsanlæg skaleres PFAPR-kravet til indsvingningsforløb og nøjagtighed med 
udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1). 
 
18/39 
 
Dok.23/13192-1 Offentlig/Public 
Hvis produktionsanlæg samplaceres med energilager- og/eller forbrugsanlæg, skal produkti-
onsanlægget bidrage til, at det samlede anlæg efter et indsvingningsforløb opnår normal ef-
fektudveksling, når driftsforholdene i PoC er tilbage i området kontinuert drift. Systembrugeren 
har til ansvar at koordinere bidrag fra de samplacerede anlæg, så det samlede anlæg følger  
kravene til indsvingningsforløb, tid og nøjagtighed. Kravene differentieres afhængigt af, om det 
samlede anlæg var i produktions- eller forbrugstilstand forud for hændelsen: 
• Produktionstilstand: NC RfG art. 20.3.a 
• Forbrugstilstand: TF 3.4.3 §12. 
 
Kravene til forbrugstilstand er angivet i bilag i Afsnit 6.2.1, og disse tilføjes til NC RfG ifm. ende-
lig implementering af krav til overplanting og samplacering. 
  
Hvis produktionsanlægget udelukkende samplaceres med et energilageranlæg, anvendes 
PFAPR-karakteristikken fra NC RfG art. 20.3.a for indsvingningsforløb, tid og nøjagtighed ved 
både produktions- og forbrugstilstand af det samlede anlæg. 
 
Energilageranlæg: 
Energilageranlæg skal have PFAPR-egenskaber jf. kravene i TF 3.3.1 §124. For overplantede 
energilageranlæg skaleres PFAPR-kravet til indsvingningsforløb og nøjagtighed med udveks-
lingskapacitet (PnG3 og Pnd3) frem for installeret effekt (PnG1 og PnD1). 
 
Hvis energilageranlæg samplaceres med produktions- og/eller forbrugsanlæg, skal energilager-
anlægget bidrage til, at det samlede anlæg efter et indsvingningsforløb opnår normal effektud-
veksling, når driftsforholdene i PoC er tilbage i området kontinuert drift. Systembrugeren har til 
ansvar at koordinere bidrag fra de samplacerede anlæg, så det samlede anlæg følger kravene 
til indsvingningsforløb, tid og nøjagtighed. Kravene differentieres afhængigt af, om det samlede 
anlæg var i produktions- eller forbrugstilstand forud for hændelsen: Bemærk, at PFAPR-kravet 
er ens i NC RfG art. 20.3.a og TF 3.3.1 §124.  
• Produktionstilstand: TF 3.3.1 §124 
• Forbrugstilstand: TF 3.4.3 §12. 
 
Kravene til forbrugstilstand er angivet i bilag i Afsnit 6.2.1, og disse tilføjes til TF 3.3.1 ifm. en-
delig implementering af krav til overplanting og samplacering. 
 
Hvis energilageranlægget udelukkende samplaceres med et produktionsanlæg, anvendes 
PFAPR-karakteristikken fra TF 3.3.1 §124 for indsvingningsforløb, tid og nøjagtighed ved både 
produktions- og forbrugstilstand af det samlede anlæg. 
 
Forbrugsanlæg: 
Forbrugsanlæg skal have PFAPR-egenskaber jf. kravene i TF 3.4.3 §12. For overplantede for-
brugsanlæg skaleres PFAPR-kravet til indsvingningsforløb og nøjagtighed med udvekslingskapa-
citet (PnD3) frem for installeret effekt (PnD1). 
 
Hvis forbrugsanlæg samplaceres med produktions- og/eller energilageranlæg, skal forbrugsan-
lægget bidrage til, at det samlede anlæg efter et indsvingningsforløb opnår normal effektudveks-
ling, når driftsforholdene i PoC er tilbage i området kontinuert drift. Systembrugeren har til an-
svar at koordinere bidrag fra de samplacerede anlæg, så det samlede anlæg følger kravene til  
indsvingningsforløb, tid og nøjagtighed. Kravene differen tieres afhængigt af,  om det samlede 
anlæg var i produktions- eller forbrugstilstand forud for hændelsen:  
• Produktionstilstand: NC RfG art. 20.3.a 
19/39 
 
Dok.23/13192-1 Offentlig/Public 
• Forbrugstilstand: TF 3.4.3 §12. 
  
Kravene til produktionstilstand er angivet i bilag i afsnit 6.2.2, disse tilføjes til TF 3.4.3 ifm. en-
delig implementering af krav til overplanting og samplacering. 
 
5.9 Rate of Change of Frequency (ROCOF) 
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg ensrettes kravene til 
ROCOF. Herved sikres det, at der ikke forekommer udkobling af dele af det samlede anlæg, når 
det drives i normaldriftsområdet. 
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 13.1.b 
• Energilageranlæg: TF 3.3.1 §11 
• Forbrugsanlæg: NC DC art. 28.2.k. 
 
Produktionsanlæg: 
Ingen ændring. 
 
Energilageranlæg: 
Ingen ændring. 
 
Forbrugsanlæg: 
Hvis forbrugsanlægget samplaceres med energilager- og/eller, så pålægges det forbrugsanlæg-
get at beregne ROCOF-frekvensændringen som gennemsnittet over en periode på 200 ms jf. 
NC RfG 13.1.b. 
 
5.10 Limited frequency sensitivity mode – Overfrequency (LFSM-O)  
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg pålægges krav til LFSM-
O for det samlede anlæg i PoC med baggrund i udvekslingskapaciteten. Kravet har til formål at 
sikre, at det samlede anlæg yder et koordineret frekvensrespons for at understøtte det kollek-
tive elsystem ifm. overfrekvens.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 13.2 
• Energilageranlæg: TF 3.3.1 §23 
• Forbrugsanlæg: Ingen eksisterende krav. 
 
Produktionsanlæg: 
Produktionsanlæggets egenskaber til LFSM-O skal overholde kravene i NC RfG art. 13.2. For 
overplantede produktionsanlæg ændres kravet således, at den responderende aktive effekt 
skaleres med udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1) . 
 
Hvis produktionsanlægget samplaceres med energilager- og/eller forbrugsanlæg, tillades det, 
at de samplacerede anlæg bidrager til opfyldelse af LFSM-O-kravet under forudsætning af, at 
produktionsanlæggets og de bidragende anlægs aktiv effekt-bidrag koordineres således, at bi-
draget af aktiv effekt for det samlede anlæg opfylder den krævede karakteristik jf. NC RfG  art. 
13.2. Det samlede anlæg må ikke overskride P
nd3. 
 
Energilageranlæg: 
20/39 
 
Dok.23/13192-1 Offentlig/Public 
Energilageranlæggets egenskaber til LFSM-O skal overholde kravene i TF 3.3.1 §22. For over-
plantede energilageranlæg ændres kravet således, at den responderende aktive effekt skaleres 
med udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1). Det samlede anlæg må ikke 
overskride Pnd3. 
 
Hvis energilageranlægget samplaceres med et produktions- og/eller forbrugsanlæg, tillades 
det, at de samplacerede anlæg bidrager til opfyldelse af LFSM-O-kravet under forudsætning af, 
at energilageranlæggets og de bidragende anlægs aktive effekt koordineres således, at bidra-
get af aktiv effekt for det samlede anlæg opfylder den krævede karakteristik jf. TF 3.3.1 §23. 
Det samlede anlæg må ikke overskride Pnd3. 
 
Forbrugsanlæg: 
Forbrugsanlægget må ikke påvirke den udvekslede strøm under et hændelsesforløb, så det 
samlede anlægs respons afviger fra den påkrævede LFSM-O-karakteristik. 
 
Hvis systembrugeren vælger at udnytte forbrugsanlæggets egenskaber som bidrag til opfyl-
delse af energilageranlæggets og/eller produktionsanlæggets påkrævede LFSM-O-
karakteristikker, er det systembrugerens ansvar, at der implementeres en teknisk løsning til ko-
ordinering af det samlede anlægs LFSM-O-respons i PoC. 
 
5.11 Limited frequency sensitivity mode – Underfrequency (LFSM-U)  
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg pålægges krav til LFSM-
U for anlæggene med baggrund i udvekslingskapaciteten i PoC. Kravet har til formål at sikre, at 
det samlede anlæg yder et koordineret frekvensrespons for at understøtte det kollektive elsy-
stem ifm. underfrekvens.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.2.c 
• Energilageranlæg: TF 3.3.1 §52 
• Forbrugsanlæg: TF 3.4.3 §13. 
 
Produktionsanlæg: 
Produktionsanlæggets egenskaber til LFSM-U skal overholde kravene i NC RfG art. 15.2.c. For 
overplantede produktionsanlæg ændres kravet således, at den responderende aktive effekt 
skaleres med udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1). Det samlede anlæg 
må ikke overskride PnG3. 
 
Hvis produktionsanlægget samplaceres med et energilageranlæg, tillades det, at det samplace-
rede anlæg bidrager til opfyldelse af LFSM-U-kravet under forudsætning af, at produktionsan-
læggets og det bidragende anlægs aktiv effekt-bidrag koordineres således, at bidraget af aktiv 
effekt for det samlede anlæg opfylder den krævede karakteristik jf. NC RfG art. 15.2.c. Det 
samlede anlæg må ikke overskride P
nG3. 
 
Hvis produktionsanlægget samplaceres med et forbrugsanlæg, fastholdes statikken jf. kravet 
fra NC RfG art. 15.2.c uagtet den ændring af aktiv effekt i PoC, som forbrugsanlæggets LFSM-U-
respons måtte medføre. 
 
Energilageranlæg: 
Energilageranlæggets egenskaber til LFSM-U skal overholde kravene i TF 3.3.1 §52. For over-
plantede energilageranlæg ændres kravet således, at den responderende aktive effekt skaleres 
21/39 
 
Dok.23/13192-1 Offentlig/Public 
med udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1). Det samlede anlæg må ikke 
overskride PnG3. 
 
Hvis energilageranlægget samplaceres med et produktionsanlæg, kræves det, at aktiv effekt-
bidraget fra energilageranlægget koordineres med produktionsanlæggets således, at summen 
af aktiv effekt målt i PoC følger den krævede karakteristik fra kravene i NC RfG art. 15.2.c og TF 
3.3.1 §13. Det samlede anlæg må ikke overskride PnG3. 
 
Hvis energilageranlægget samplaceres med et forbrugsanlæg, fastholdes statikken jf. kravet fra 
NC RfG art. 15.2.c uagtet den ændring af aktiv effekt i PoC, som forbrugsanlæggets LFSM-U re-
spons måtte medføre. 
 
Forbrugsanlæg: 
Hvis forbrugsanlægget samplaceres med produktions- og/eller energilageranlæg, skal forbrugs-
anlæggets egenskaber til LFSM-U overholde kravene i TF 3.4.3 §13. Det samlede anlæg må ikke 
overskride PnG3. 
 
5.12 Low Frequency Demand Disconnection (LFDD)  
I forbindelse med en hændelse i det kollektive elsystem, som resulterer i ekstrem underfre-
kvens, skal det individuelle forbrugsanlæg, som indgår i samplacering, kunne understøtte elsy-
stemet. Det er systembrugerens ansvar, at det samlede anlæg ikke overskrider udvekslingska-
paciteten (P
nG3).  
 
Eksisterende krav 
• Produktionsanlæg: Ingen eksisterende krav 
• Energilageranlæg: Ingen eksisterende krav 
• Forbrugsanlæg: NC DC art. 19.1. 
 
Produktionsanlæg: 
Hvis produktionsanlægget samplaceres med et forbrugsanlæg, pålægges produktionsanlægget 
at regulere udvekslingen af aktiv effekt i PoC i situationer, hvor forbrugsanlægget aflastes som 
følge af kravene fra NC DC art. 19.1, således, at PnG3 ikke overskrides. 
 
Energilageranlæg: 
Hvis energilageranlægget samplaceres med et forbrugsanlæg, pålægges energilageranlægget 
at regulere udvekslingen af aktiv effekt i PoC i situationer, hvor forbrugsanlægget aflastes som 
følge af kravene fra NC DC art. 19.1, således, at PnG3 ikke overskrides. 
 
Forbrugsanlæg: 
Hvis forbrugsanlæg samplaceres med produktions- og/eller energilageranlæg, skal forbrugsan-
lægget kunne automatisk aflaste aktuelt aktiv effekt forbrug i de definerede trin jf. NC DC art 
19.1. Hvis det samlede anlægs udveksling af aktiv effekt med det kollektive elsystem bliver lig 
PnG3 ifm. aflastning kræves der ikke yderligere aktivering af automatisk aflastningstrin. 
 
5.13 Manuel aflastning  
I forbindelse med overhængende risiko for netsammenbrud, under netsammenbrud og under 
genopbygning kan Energinet aktivere manuel aflastning. Dette skal fortsat være muligt ved in-
dividuelle forbrugsanlæg, som indgår i samplacering. Det er produktions- og/eller energilager-
anlæggets ansvar at sikre, at udvekslingskapaciteten (P
nG3) ikke overskrides. 
22/39 
 
Dok.23/13192-1 Offentlig/Public 
 
Eksisterende krav 
• Produktionsanlæg: Ingen eksisterende krav 
• Energilageranlæg: TF 3.3.1 §74 
• Forbrugsanlæg: TF 3.4.2 §2. 
 
Produktionsanlæg: 
Hvis produktionsanlægget samplaceres med et forbrugsanlæg, pålægges produktionsanlægget 
at regulere udvekslingen af aktiv effekt i PoC i situationer, hvor forbrugsanlægget aflastes som 
følge af kravene fra TF 3.4.2 §2, således, at PnG3 ikke overskrides. 
 
Hvis produktionsanlægget samplaceres med et energilageranlæg, pålægges produktionsanlæg-
get at regulere udvekslingen af aktiv effekt i PoC i situationer, hvor energilageranlægget afla-
stes som følge af kravene fra TF 3.3.1 §74, således, at PnG3 ikke overskrides. 
 
Energilageranlæg: 
Energilageranlæg pålægges kravene fra TF 3.3.1 §74.  
 
Hvis energilageranlægget samplaceres med et forbrugsanlæg, pålægges energilageranlægget 
at regulere udvekslingen af aktiv effekt i PoC i situationer, hvor forbrugsanlægget aflastes som 
følge af kravene fra TF 3.4.2 §2, således, at P
nG3 ikke overskrides. 
 
Forbrugsanlæg: 
Ingen ændring.  
 
5.14 Begrænsning af spændingsvariationer ved spændingssætning 
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg udvides spændingssæt-
ningskrav til begrænsning af spændingsvariationer med specificering af evalueringskriteriet. 
Grænseværdierne for statisk spændingsvariation fra eksisterende krav fastholdes. Grænsevær-
dierne gælder for det samlede anlægs påvirkning af spændingen, og derfor evalueres spæn-
dingsvariationen for samplacerede anlæg som summen af spændingsændring i PoC. Grænse-
værdierne for de enkelte anlægstyper summeres ikke.  
 
Eksisterende krav 
• Produktionsanlæg: TF 3.2.7 afsnit 9.1.3 
• Energilageranlæg: TF 3.3.1 §75 
• Forbrugsanlæg: TF 3.4.3 §6. 
 
Produktionsanlæg: 
Produktionsanlæg, der samplaceres med energilager- og/eller forbrugsanlæg, skal overholde 
krav til spændingsvariationer angivet i TF 3.2.7. For samplacerede anlæg evalueres spændings-
variationen som den samlede ændring i statisk spænding før og efter spændingssætning, uan-
set om det spændingssatte udstyr tilhører produktionsanlægget, energilageranlægget, for-
brugsanlægget eller kombinationer heraf. Det er systembrugerens ansvar, at der implemente-
res en teknisk løsning til koordineret overholdelse af spændingsvariationsgrænser.  
 
Energilageranlæg: 
Energilageranlæg, der samplaceres med produktions- og/eller forbrugsanlæg, skal overholde 
krav til spændingsvariationer angivet i TF 3.3.1. For samplacerede anlæg evalueres spændings-
variationen som den samlede ændring i statisk spænding før og efter spændingssætning, 
23/39 
 
Dok.23/13192-1 Offentlig/Public 
uanset om det spændingssatte udstyr tilhører produktionsanlægget, energilageranlægget , for-
brugsanlægget eller kombinationer heraf. Det er systembrugerens ansvar, at der implemente-
res en teknisk løsning til koordineret overholdelse af spændingsvariationsgrænser. 
  
Forbrugsanlæg: 
Forbrugsanlæg, der samplaceres med energilager- og/eller produktionsanlæg, skal overholde 
krav til spændingsvariationer angivet i TF 3.4.3. For samplacerede anlæg evalueres spændings-
variationen som den samlede ændring i statisk spænding før og efter spændingssætning, uan-
set om det spændingssatte udstyr tilhører produktionsanlægget, energilageranlægget , for-
brugsanlægget eller kombinationer heraf. Det er systembrugerens ansvar, at der implemente-
res en teknisk løsning til koordineret overholdelse af spændingsvariationsgrænser. 
 
5.15 Power Oscillation Damping (POD) 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
skaleres kravene til POD på basis af største udvekslingskapacitet (P
nG3 eller Pnd3). 
Det kræves, at det samlede anlæg dæmper effekt-oscillationer i PoC til et vist niveau inden for 
et specificeret tidsrum for at undgå utilsigtet aktivering af beskyttelsesudstyr.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 21.3.f (anmeldt) 
• Energilageranlæg: TF 3.3.1 §121 
• Forbrugsanlæg: NC DC art. 17.2.b (anmeldt). 
 
Produktionsanlæg: 
Produktionsanlægget skal dæmpe effekt-oscillationer jf. kravene i NC RfG art. 21.3.f. For over-
plantede og/eller samplacerede produktionsanlæg ændres POD-kravet, så den påkrævet græn-
seværdi for effekt-oscillationer skaleres med største udvekslingskapacitet (PnG3 eller Pnd3) frem 
for installeret effekt (PnG1). 
 
Produktionsanlæg, der er samplacerede med energilager- og/eller forbrugsanlæg, skal i koordi-
nation med de øvrige samplacerede anlæg sikre, at POD-kravet overholdes. Det er systembru-
gerens ansvar, at der implementeres en teknisk løsning til koordinering af det samlede anlægs 
POD. 
  
Energilageranlæg: 
Energilageranlæg skal dæmpe effekt-oscillationer jf. kravene i TF 3.3.1. For overplantede og/el-
ler samplacerede energilageranlæg ændres POD-kravet, så den påkrævet grænseværdi for ef-
fekt-oscillationer skaleres med største udvekslingskapacitet (PnG3 eller Pnd3) frem for installeret 
effekt (PnG1). 
 
Energilageranlæg, der er samplacerede med produktions - og/eller forbrugsanlæg, skal i koor-
dination med de øvrige samplacerede anlæg sikre, at POD-kravet overholdes. Det er system-
brugerens ansvar, at der implementeres en teknisk løsning til koordinering af det samlede an-
lægs POD. 
  
Forbrugsanlæg: 
Forbrugsanlæg skal dæmpe effekt-oscillationer jf. kravene i NC DC art. 17.2.b. For overplan-
tede og/eller samplacerede forbrugsanlæg ændres POD-kravet, så den påkrævet grænseværdi 
for effekt-oscillationer skaleres med største udvekslingskapacitet (PnG3 eller Pnd3) frem for in-
stalleret effekt (PnG1). 
24/39 
 
Dok.23/13192-1 Offentlig/Public 
 
Forbrugsanlæg, der er samplacerede med produktions- og/eller energilageranlæg, skal i koor-
dination med de øvrige samplacerede anlæg sikre, at POD-kravet overholdes. Det er system-
brugerens ansvar, at der implementeres en teknisk løsning til koordinering af det samlede an-
lægs POD. 
 
5.16 Elkvalitet 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
pålægges krav til elkvalitet, hvor emissionsgrænser udmøntes med baggrund i udvekslingska-
paciteten (P
nG3 og/eller Pnd3). Ligeledes pålægges krav om, at samplacerede anlæg overholder 
de fastsatte emissionsgrænser som et samlet anlæg. Kravet har til formål at sikre, at de sam-
placerede anlægs elkvalitetspåvirkning koordineres, så emissionsgrænserne fastsat af Energi-
net overholdes, og driftsforstyrrende forringelse af elkvaliteten i det kollektive elsystem  derved 
forhindres. 
  
Eksisterende krav 
• Produktionsanlæg: TF 3.2.7 
• Energilageranlæg: TF 3.3.1 §73 (henvisning til TF 3.2.7) 
• Forbrugsanlæg: NC DC art. 20 (henvisning til NC DC bilag 1.E). 
  
Produktionsanlæg: 
Produktionsanlægget pålægges at efterleve kravene til elkvalitet som beskrevet i TF 3.2.7. For 
et overplantet produktionsanlæg vil emissionsgrænser blive fastsat af Energinet på basis af ud-
vekslingskapaciteten (PnG3). For et samplaceret produktionsanlæg vil emissionsgrænser blive 
fastsat af Energinet på basis af den største udvekslingskapacitet (PnG3 eller Pnd3). 
 
Hvis produktionsanlægget samplaceres med energilager- og/eller forbrugsanlæg, fastsættes et 
sæt af elkvalitetskrav, som det samlede anlæg skal overholde. Det er systembrugerens ansvar, 
at der implementeres en teknisk løsning til koordineret overholdelse af det samlede anlægs el-
kvalitets-emissionsgrænser.  
 
Det påkræves, at eftervisning af kravene til elkvalitet foretages som beskrevet i TF 3.2.7, men 
udført som en samlet vurdering for det samlede anlæg. Energinet godkender både systembru-
gerens beregnings- og målemetoder og den endelige dokumentation for verifikation af samt-
lige elkvalitetsparametre. 
  
Energilageranlæg: 
Energilageranlæg vil blive pålagt at efterleve kravene til elkvalitet som beskrevet i TF 3.2.7 (TF 
3.3.1 henviser hertil). For et overplantet energilageranlæg vil emissionsgrænser blive fastsat af 
Energinet på basis af udvekslingskapaciteten (PnG3). For et samplaceret energilageranlæg vil 
emissionsgrænser blive fastsat af Energinet på basis af den største udvekslingskapacitet (P nG3 
eller Pnd3). 
 
Hvis energilageranlægget samplaceres med produktions- og/eller forbrugsanlæg, fastsættes et 
sæt af elkvalitetskrav, som det samlede anlæg i fællesskab skal overholde. Det er systembruge-
rens ansvar, at der implementeres en teknisk løsning til koordineret overholdelse af det fælles 
sæt af elkvalitets-emissionsgrænser.  
 
25/39 
 
Dok.23/13192-1 Offentlig/Public 
Det påkræves, at eftervisning af kravene til elkvalitet foretages som beskrevet i TF 3.2.7, men 
udført som en samlet vurdering af alle samplacerede anlæg. Energinet godkender både sy-
stembrugerens beregnings- og målemetoder og den endelige dokumentation for verifikation af 
samtlige elkvalitetsparametre.  
  
Forbrugsanlæg: 
Forbrugsanlæg vil blive pålagt at efterleve kravene til elkvalitet som beskrevet i NC DC bilag 
1.E. For et overplantet forbrugsanlæg vil emissionsgrænser blive fastsat af Energinet på basis af 
udvekslingskapaciteten (Pnd3). For et samplaceret forbrugsanlæg vil emissionsgrænser blive 
fastsat af Energinet på basis af den største udvekslingskapacitet (PnG3 eller Pnd3). 
 
Hvis forbrugsanlægget samplaceres med energilager- og/eller produktionsanlæg, fastsættes et 
sæt af elkvalitetskrav, som det samlede anlæg i fællesskab skal overholde. Det er systembruge-
rens ansvar, at der implementeres en teknisk løsning til koordineret overholdelse af det fælles 
sæt af elkvalitets-emissionsgrænser.   
 
Det påkræves, at eftervisning af kravene til elkvalitet foretages som beskrevet i NC DC bilag 1.E 
for krav til forvrængningsbidrag baseret på baggrundstøjsmåling. Eftervisning skal udføres som 
en samlet vurdering af alle samplacerede anlæg. Energinet godkender både systembrugerens 
beregnings- og målemetoder og den endelige dokumentation for verifikation af samtlige elkva-
litetsparametre. Det bemærkes, at der er samme krav til eftervisning af elkvalitet i NC DC bilag 
1.E og TF 3.2.7 (gældende for produktions- og energilageranlæg).  
 
5.17 Aktiv effekt-referencepunkt 
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg pålægges kravet til kon-
trollerbarhed for aktiv effekt-referencepunkt jf. NC RfG art. 15.2.a for det samlede anlæg i PoC 
med baggrund i den største udvekslingskapacitet (P
nG3 eller Pnd3). Kravet har til formål at sikre, 
at det samlede anlæg, ved modtagelse af ordre om ændring af aktiv effekt referencepunkt,  
yder et koordineret respons, som understøtter det kollektive elsystems driftsmæssige behov.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.2.a 
• Energilageranlæg: TF 3.3.1 §9 
• Forbrugsanlæg: Ingen eksisterende krav. 
 
Produktionsanlæg: 
Produktionsanlæggets reguleringsegenskaber skal efterleve krav til at opnå referencepunkter 
jf. NC RfG art. 15.2.a. Kravet indeholder specifikation af opløsning og tolerancer for nøjagtighe-
den af reguleringen. Afvigelser fra tolerancen tillades af hensyn til tilgængeligheden af primær 
energi. Ved opregulering efter den primære energi vender tilbage, skal anlægget følge kravet 
til rampehastigheder jf. NC RfG art. 15.6.e uddybet i Afsnit 5.18 for samplacering og overplan-
ting. For overplantede produktionsanlæg skaleres kravet til aktiv effekt kontrollerbarhed og 
sætpunkter med udvekslingskapaciteten (PnG3) frem for installeret effekt (PnG1).  For samplace-
rede produktionsanlæg skaleres kravet til aktiv effekt kontrollerbarhed og sætpunkter med den 
største udvekslingskapacitet (PnG3 eller Pnd3) frem for installeret effekt (PnG1).  
 
Hvis produktionsanlægget samplaceres med energilager- og/eller forbrugsanlæg, tillades det, 
at de samplacerede anlæg bidrager til opfyldelse af kravet til kontrollerbarhed for aktiv effekt-
referencepunkt under forudsætning af, at produktionsanlægget og alle bidragende anlæg ko-
ordineres og styres af én fælles parkregulator. Ved manglende tilgængelighed af primær energi 
26/39 
 
Dok.23/13192-1 Offentlig/Public 
skal udvekslingen til det kollektive elsystem prioriteres over egetforbrug for samplacerede 
energilager- og/eller forbrugsanlæg.  
 
Energilageranlæg: 
Energilageranlæggets reguleringsegenskaber skal efterleve krav til at opnå referencepunkter jf. 
TF 3.3.1 §9. For overplantede og/eller samplacerede energilageranlæg skaleres kravet til tole-
rance og reguleringshastighed af aktiv effekt-sætpunkter med den største udvekslingskapacitet 
(PnG3 eller Pnd3) frem for installeret effekt (PnG1).  
 
Hvis energilageranlægget samplaceres med et forbrugsanlæg, tillades det, at det samplacerede 
anlæg bidrager til opfyldelse af kravet til kontrollerbarhed for at opnå referencepunkter for ak-
tiv effekt i PoC under forudsætning af, at alle bidragende anlæg koordineres og styres af én 
fælles parkregulator. 
 
Energilageranlæg, der samplaceres med et produktionsanlæg, må ikke bevirke, at den udveks-
lede aktive effekt i PoC afviger fra kravene til opløsning og tolerance fastsat for produktion san-
lægget. Hvis energilageranlægget bidrager til opfyldelse af produktionsanlæggets påkrævede 
kontrollerbarhed og opfyldelse af sætpunkt for aktiv effekt i PoC, kræves det, at produktions-
anlægget og alle bidragende anlæg koordineres og styres af én fælles parkregulator.  
 
Forbrugsanlæg: 
Forbrugsanlæg, der samplaceres med energilager- og/eller produktionsanlæg, har ikke krav 
vedrørende kontrollerbarhed for at opnå referencepunkter for aktiv effekt i PoC. Forbrugsan-
lægget må bidrage til opfyldelse af energilageranlæggets og/eller produktionsanlæggets på-
krævede krav til aktiv effekt-regulering i PoC under forudsætning af, at alle bidragende anlæg 
koordineres og styres af én fælles parkregulator. 
 
5.18 Aktiv effekt-reguleringsrampe 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
pålægges krav til minimums- og maksimumsgradienter for ændring af aktiv effekt i PoC for det 
samlede anlæg med baggrund i den største udvekslingskapacitet (P
nG3 eller Pnd3). Kravet har til 
formål at sikre, at det samlede anlægs regulering af aktiv effekt er underlagt rampebegræns-
ninger, så den aktive effekt-regulering gennemføres på hensigtsmæssig tid ift. det kollektiv el-
systems drift. 
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.6.e 
• Energilageranlæg: TF 3.3.1 §32 
• Forbrugsanlæg: TF 3.4.3 §4. 
 
Produktionsanlæg: 
Produktionsanlægget skal have minimum og maksimum rampebegrænsninger for ændring af 
aktiv effekt ved op- og nedregulering jf. NC RfG art. 15.6.e. Reguleringen skal foregå lineært i 
reguleringsperioden på 1 minut under hensyntagen til energikildens teknologiske karakteri-
stika. For overplantede og/eller samplacerede produktionsanlæg skaleres kravet til aktiv effekt-
rampebegrænsninger med den største udvekslingskapacitet (PnG3 eller Pnd3) frem for installeret 
effekt (PnG1).  
 
Hvis produktionsanlægget regulerer aktiv effekt samtidig med samplacerede energilager- 
og/eller forbrugsanlæg, må produktionsanlægget ikke rampe aktiv effekt i en sådan grad, at 
27/39 
 
Dok.23/13192-1 Offentlig/Public 
ændring af aktiv effekt-udveksling mellem det kollektive elsystem og det samlede anlæg over-
stiger rampebegrænsningens maksimalværdi i PoC. Det er systembrugerens ansvar, at der im-
plementeres en teknisk løsning, som koordinerer de samplacerede anlægs aktiv effekt-regule-
ring således, at det samlede anlæg overholder rampebegrænsningens maksimalværdi.  
 
Energilageranlæg: 
Energilageranlæg skal have minimum og maksimum rampebegrænsninger for ændring af aktiv 
effekt ved op- og nedregulering jf. TF 3.3.1 §32. Reguleringen skal foregå lineært eller tilnær-
met lineært ved en trinfunktion i reguleringsperioden på 1 minut. For overplantede og/eller 
samplacerede energilageranlæg skaleres kravet til ændringer af aktiv effekt med den største 
udvekslingskapacitet (PnG3 eller Pnd3) frem for installeret effekt (PnG1). 
 
Hvis energilageranlægget regulerer aktiv effekt samtidig med samplacerede produktions- 
og/eller forbrugsanlæg, så må energilageranlægget ikke rampe aktiv effekt i en sådan grad, at 
ændring af aktiv effekt-udveksling mellem det kollektive elsystem og det samlede anlæg over-
stiger rampebegrænsningens maksimalværdi i PoC. Det er systembrugerens ansvar, at der im-
plementeres en teknisk løsning, som koordinerer de samplacerede anlægs aktiv effekt -regule-
ring således, at det samlede anlæg overholder rampebegrænsningens maksimalværdi.  
 
Forbrugsanlæg: 
Forbrugsanlæg skal have maksimum rampebegrænsninger for ændring af aktiv effekt ved op- 
og nedregulering jf. TF 3.4.3 §4. Reguleringen skal foregå lineært eller tilnærmet lineært ved en 
trinfunktion i reguleringsperioden på 1 minut. For overplantede og/eller samplacerede for-
brugsanlæg skaleres kravet til ændringer af aktiv effekt med den største udvekslingskapacitet 
(PnG3 eller Pnd3) frem for installeret effekt (PnD1). 
 
Hvis forbrugsanlægget regulerer aktiv effekt samtidig med samplacerede energilager- og/eller 
produktionsanlæg, må forbrugsanlægget ikke rampe aktiv effekt i sådan grad, at ændring af ak-
tiv effekt-udveksling mellem det kollektive elsystem og det samlede anlæg overstiger rampebe-
grænsningens maksimalværdi i PoC. Det er systembrugerens ansvar, at der implementeres en 
teknisk løsning, som koordinerer de samplacerede anlægs aktiv effekt-regulering således, at 
det samlede anlæg overholder rampebegrænsningens maksimalværdi.  
 
5.19 Reaktiv effekt-egenskaber 
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg pålægges kravet til re-
aktiv effekt-egenskaber for det samlede anlæg i PoC med baggrund i udvekslingskapacitete n. 
Det tillades ikke, at anlægget kortvarigt nedprioriterer aktiv effekt for at overholde krav til re-
aktiv effekt-egenskaber i normaldriftsområdet.  
Ved overplanting fastsættes de påkrævede reaktiv effekt-egenskaber på basis af udvekslingska-
paciteten (P
nG3).  
Kravet har til formål at sikre, at det samlede anlæg har reaktiv effekt-egenskaber tilsvarende et 
selvstændigt produktionsanlæg med samme aftalte udvekslingskapacitet PnG3. 
  
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 21.3.b – 21.3.c 
• Energilageranlæg: TF 3.3.1 §123, §126 og §127 
• Forbrugsanlæg: NC DC art. 15.1.a. 
  
Produktionsanlæg: 
28/39 
 
Dok.23/13192-1 Offentlig/Public 
Produktionsanlægget skal have reaktiv effekt-egenskaber jf. kravene i NC RfG art. 21.3.b og art. 
21.3.c. For overplantede produktionsanlæg ændres reaktiv effekt-kravet, så den påkrævede 
reaktive effekt skaleres med udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1). 
  
Samplacerede produktionsanlæg skal opfylde kravene til reaktiv effekt-egenskaber, uagtet om 
det samlede anlæg er i produktions- eller forbrugstilstand. Hvis produktionsanlægget sampla-
ceres med et energilageranlæg, kræves det, at de reaktive effekt-egenskaber for energilager-
anlægget og produktionsanlægget koordineres således, at det samlede anlæg kan levere den 
påkrævede reaktive effekt i PoC jf. NC RfG art. 21.3.b-c og TF 3.3.1 §123, §126 og §127. Be-
mærk, at de påkrævede reaktiv effekt-egenskaber er ens jf. NC RfG og TF 3.3.1.    
 
Hvis produktionsanlægget regulerer reaktiv effekt samtidig med samplacerede forbrugsanlæg 
og/eller energilageranlæg, må produktionsanlægget ikke forårsage, at den reaktive effekt-ud-
veksling mellem det kollektive elsystem og det samlede anlæg afviger fra det sætpunkt, som er 
påkrævet heraf i PoC. Det er systembrugerens ansvar, at der implementeres en teknisk løsning, 
som koordinerer de samplacerede anlægs reaktiv effekt-udveksling således, at det samlede an-
læg leverer den påkrævede reaktive effekt.  
  
Det accepteres, at reaktiv effekt-egenskaber kan begrænses jf. NC RfG art. 21.3.c hvis et redu-
ceret antal af produktionsanlæggets enheder er i drift grundet opstart og nedlukning som funk-
tion af primær energi, vedligehold eller fejl. Begrænsning af reaktiv effekt-egenskaber skal 
følge den aktuelle produktion frem for den udvekslede effekt.  
  
Energilageranlæg: 
Energilageranlæg skal have reaktiv effekt-egenskaber jf. kravene i TF 3.3.1 §123, §126 og §127. 
For overplantede energilageranlæg ændres reaktiv effekt-kravet, så den påkrævede reaktive 
effekt skaleres med udvekslingskapacitet (PnG3) frem for installeret effekt (PnG1). 
  
Samplacerede energilageranlæg skal opfylde kravene til reaktiv effekt-egenskaber, uagtet om 
det samlede anlæg er i produktions- eller forbrugstilstand.  
 
Hvis energilageranlægget samplaceres med et produktionsanlæg, kræves det, at de reaktive 
effekt-egenskaber for energilageranlægget og produktionsanlægget koordineres således, at 
det samlede anlæg kan levere den påkrævede reaktive effekt i PoC jf. NC RfG art. 21.3.b-c og 
TF 3.3.1 §123, §126 og §127. Bemærk, at de påkrævede reaktiv effekt-egenskaber er ens jf. NC 
RfG og TF 3.3.1. Det accepteres, at reaktiv effekt-egenskaber kan begrænses jf. NC RfG art. 
21.3.c hvis et reduceret antal af produktionsanlæggets enheder er i drift grundet opstart og 
nedlukning som funktion af primær energi, vedligehold eller fejl.  
 
Hvis energilageranlægget regulerer reaktiv effekt samtidig med samplacerede produktions- 
og/eller forbrugsanlæg, så må energilageranlægget ikke forårsage, at den reaktive effekt-ud-
veksling mellem det kollektive elsystem og det samlede anlæg afviger fra det sætpunkt, som er 
påkrævet heraf i PoC. Det er systembrugerens ansvar, at der implementeres en teknisk løsning, 
som koordinerer de samplacerede anlægs reaktiv effekt-udveksling således, at det samlede an-
læg leverer den påkrævede reaktive effekt.  
  
Forbrugsanlæg: 
Forbrugsanlæg, der samplaceres med energilager- og/eller produktionsanlæg i et samlet anlæg 
med PnG3 større end 0 MW, skal fastholde en effektfaktor cos(phi) = 1,00 i PoC. Det tillades dog, 
at forbrugsanlægget bidrager til opfyldelse af energilageranlæggets og/eller 
29/39 
 
Dok.23/13192-1 Offentlig/Public 
produktionsanlæggets påkrævede reaktive effekt. Det er systembrugerens ansvar, at der im-
plementeres en teknisk løsning, som koordinerer de samplacerede anlægs reaktiv effekt -ud-
veksling således, at det samlede anlæg leverer den påkrævede reaktive effekt.   
 
5.20 Reaktiveffektregulering 
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg pålægges kravet til re-
aktiv effekt-regulering jf. NC RfG art. 21.3.d for det samlede anlæg i PoC, herunder kravene til 
spændingsreguleringstilstand, reaktiv effekt-reguleringstilstand og effektfaktorreguleringstil-
stand. Kravet har til formål at sikre, at det samlede anlæg kan yde reaktiv effekt-regulering til-
svarende et selvstændigt produktions- eller energilageranlæg med samme aftalte udvekslings-
kapacitet P
nG3. 
  
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 21.3.d 
• Energilageranlæg: TF 3.3.1 §130-132 
• Forbrugsanlæg: Ingen eksisterende krav. 
  
Produktionsanlæg: 
Produktionsanlægget skal efterleve kravene til reaktiv effekt-regulering angivet i NC RfG art. 
21.3.d. 
  
Hvis produktionsanlægget samplaceres med energilager- og/eller forbrugsanlæg, tillades det, 
at de samplacerede anlæg bidrager til produktionsanlæggets reaktiv effekt-regulering. Dette 
tillades under forudsætning af, at alle bidragende anlægs reaktiv effekt-bidrag koordineres og 
styres af én fælles parkregulator. 
 
Energilageranlæg: 
Energilageranlæg skal efterleve kravene reguleringsfunktioner for reaktiv effekt og spænding 
angivet i TF 3.3.1.  
 
Hvis energilageranlægget samplaceres med et forbrugsanlæg, tillades det, at det samplacerede 
anlæg bidrager funktionelt til energilageranlæggets reaktiv effekt-regulering. Dette tillades un-
der forudsætning af, at alle bidragende anlægs reaktiv effekt-bidrag koordineres og styres af én 
fælles parkregulator 
  
Hvis energilageranlægget samplaceres med et produktionsanlæg, skal energilageranlægget en-
ten drives i reaktiv effekt-reguleringstilstand med sætpunkt på 0 Mvar eller bidrage funktionelt 
til produktionsanlæggets reaktiv effekt-regulering. Hvis energilageranlægget bidrager til pro-
duktionsanlæggets reaktiv effekt-regulering, kræves det, at alle bidragende anlægs reaktiv ef-
fekt-bidrag koordineres og styres af én fælles parkregulator. 
 
Forbrugsanlæg: 
Forbrugsanlæg, der samplaceres med energilager- og/eller produktionsanlæg, må bidrage til 
opfyldelse af energilageranlæggets og/eller produktionsanlæggets påkrævede reaktiv effekt-
regulering under forudsætning af, at alle bidragende anlægs reaktiv effekt-bidrag koordineres 
og styres af én fælles parkregulator. 
 
30/39 
 
Dok.23/13192-1 Offentlig/Public 
5.21 Simuleringsmodel 
Ved samplacering af produktions-, forbrugs, og/eller energilageranlæg kræves samlede simule-
ringsmodeller for det samplacerede anlæg (stationær, RMS, EMT og harmonisk). Kravene til si-
muleringsmodeller tager udgangspunkt i kravene til de individuelle anlæg samt øvrige behov 
for samplacerede anlæg. Kravet har til formål at sikre, at Energinet modtager simuleringsmo-
deller for det samlede anlæg, som er repræsentative for anlæggets systemmæssige påvirkning 
på det kollektiv elsystem. Sådanne modeller er nødvendige for, at Energinet kan gennemføre 
net- og systemanalyser med henblik på planlægning, design og drift af det kollektive elsystem.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.6.c.i uddybet i NC RfG Bilag 1.B (anmeldt) 
• Energilageranlæg: TF 3.3.1 §78 (henviser til NC RfG og NC DC) 
• Forbrugsanlæg: NC DC art. 21.2 uddybet i NC DC Bilag 1.D (anmeldt). 
 
Produktionsanlæg: 
Produktionsanlægget skal følge alle kravene til simuleringsmodeller jf. NC RfG art. 15.6.c.i. ud-
dybet i NC RfG Bilag 1.B, med undtagelse af følgende ændringer og tilføjelser. Simuleringsmo-
deller for overplantede produktionsanlæg skal kunne repræsentere anlæggets tekniske egen-
skaber samt driftsforhold forbundet med overplantning og samplacering. For et overplantet 
produktionsanlæg evalueres nøjagtighedskrav til simuleringsmodellerne på basis af udveks-
lingskapaciteten (PnG3) frem for installeret effekt (PnG1). Simuleringsmodellerne for et overplan-
tet produktionsanlæg skal kunne afspejle interne hændelser, som resulterer i væsentlig æn-
dring af aktiv effekt evalueret i PoC. Systembrugeren er ansvarlig for at redegøre for disse hæn-
delser. Energinet vurderer, om omfanget er tilstrækkeligt.  
 
Eftersom tekniske krav til et overplantet anlæg fastsættes med udgangspunkt i udvekslingska-
paciteten (PnG3), vil en ændring af udvekslingskapaciteten anses som en væsentlig ændring af 
anlægget og kræve opdatering af simuleringsmodellerne.  
 
Hvis produktionsanlægget samplaceres med energilager- og/eller forbrugsanlæg, skal de sam-
placerede anlæg repræsenteres i samlede simuleringsmodeller for hver af de påkrævede mo-
deltyper (stationær, RMS, EMT og harmonisk). Hvis der er modstridende krav til simulerings-
modellerne for produktions-, energilager- og/eller forbrugsanlæg, som har indflydelse på op-
sætningen af modellen, er det systembrugerens ansvar at vælge det krav, som vil resultere i 
den mest retvisende repræsentation af det samlede anlæg. Energinet vurderer, om tilgangen 
kan godkendes. Simuleringsmodellerne skal være retvisende for produktionsanlægget og for 
det samlede anlægs respons i PoC.  
• Hvis produktionsanlægget bidrager til opfyldelse af tekniske krav til de øvrige sampla-
cerede anlæg, skal dette være inkluderet i de relevante simuleringsmodeller.  
• Hvis øvrige samplacerede anlæg bidrager til opfyldelse af tekniske krav til produkti-
onsanlægget, skal dette være inkluderet i de relevante simuleringsmodeller. 
• Hvis det samplacerede anlæg benytter fælles parkregulator og/eller hjælpeudstyr 
(STATCOM, synkronkompensator osv.), skal disse være inkluderet i de relevante simu-
leringsmodeller. 
• Simuleringsmodellerne skal være repræsentative for alle relevante driftsforhold  inklu-
sive selvstændig drift af produktionsanlægget.  
• Simuleringsmodellerne skal indeholde relevante signaler og målinger for produktions-
anlægget samt koordinering til øvrige anlæg og responset i PoC.  
31/39 
 
Dok.23/13192-1 Offentlig/Public 
• De relevante simuleringsmodeller skal indeholde alle beskyttelsesfunktioner for pro-
duktionsanlægget og det samlede anlæg, som er relevante for det samlede anlægs dy-
namiske respons i PoC.  
• Systembrugeren er ansvarlig for at sikre retvisende dokumentation og verifikation af 
simuleringsmodellerne for produktionsanlægget og det samplacerede anlægs respons 
i PoC. Dokumentation og verifikation skal inkludere angivelse af de anlæg i det sam-
placerede anlæg, som bidrager til overholdelse af tekniske krav. 
• Systembrugeren har ansvar for at aggregere simuleringsmodellerne for det samlede 
anlæg i så stor udstrækning som muligt jf. ovenstående. Energinet kan godkende afvi-
gelser fra øvrige aggregeringskrav, hvis der kan argumenteres for, at en anden aggre-
gering af simuleringsmodellerne giver et væsentlig bedre repræsentation af det sam-
lede anlægs respons.  
 
Energilageranlæg: 
Energilageranlæg skal følge kravene til simuleringsmodeller jf. TF 3.3.1 §78, som henviser til NC 
RfG Bilag 1.B og NC DC Bilag 1.D. Dette inkluderer krav til simuleringsmodeller i forbindelse 
med overplanting og samplacering.  
 
Forbrugsanlæg: 
Forbrugsanlæg skal følge alle kravene til simuleringsmodeller jf. NC DC art. 21.2 uddybet i NC 
DC Bilag 1.D, med undtagelse for følgende ændringer og tilføjelser. Simuleringsmodeller for 
overplantede forbrugsanlæg skal kunne repræsentere tekniske krav til anlægget samt driftsfor-
hold forbundet med overplantning og samplacering. For et overplantet forbrugsanlæg evalue-
res nøjagtighedskrav til simuleringsmodellerne på basis af udvekslingskapaciteten (Pnd3) frem 
for installeret effekt (PnD1). Simuleringsmodellen for det overplantede forbrugsanlæg skal 
kunne afspejle interne hændelser, som resulterer i væsentlig ændring af aktiv effekt evalueret i 
PoC. Systembrugeren er ansvarlig for at redegøre for disse hændelser. Energinet vurderer , om 
omfanget er tilstrækkeligt.  
 
Eftersom tekniske krav til et overplantet anlæg fastsættes med udgangspunkt i udvekslingska-
paciteten (P
nd3), vil en ændring af denne anses som en væsentlig ændring af anlægget og 
kræve opdatering af simuleringsmodellerne. 
 
Hvis forbrugsanlæg samplaceres med energilager- og/eller produktionsanlæg, skal de sampla-
cerede anlæg indgå i samlede simuleringsmodeller for hver af de påkrævede modeltyper (stati-
onær, RMS, EMT og harmonisk). Hvis der er modstridende krav til simuleringsmodellerne for 
produktions-, energilager- og/eller forbrugsanlæg, som har indflydelse på det samlede respons, 
er det systembrugerens ansvar at vælge det krav, som vil resultere i den mest retvisende re-
præsentation af det samlede anlæg. Energinet vurderer, om tilgangen kan godkendes. Simule-
ringsmodellerne skal være retvisende for forbrugsanlægget og for det samlede anlægs respons 
i PoC.  
• Hvis forbrugsanlægget bidrager til opfyldelse af tekniske krav til de øvrige samplace-
rede anlæg, skal dette være inkluderet i de relevante simuleringsmodeller.  
• Hvis øvrige samplacerede anlæg bidrager til opfyldelse af tekniske krav til forbrugsan-
lægget, skal dette være inkluderet i de relevante simuleringsmodeller. 
• Hvis det samplacerede anlæg benytter fælles parkregulator og/eller hjælpeudstyr 
(STATCOM, synkronkompensator osv.), skal disse være inkluderet i de relevante simu-
leringsmodeller. 
• Simuleringsmodellerne skal være repræsentative for alle relevante driftsforhold  inklu-
sive selvstændig drift af forbrugsanlægget.  
32/39 
 
Dok.23/13192-1 Offentlig/Public 
• Simuleringsmodellerne skal indeholde relevante signaler og målinger for forbrugsan-
lægget samt signaler til koordinering med øvrige anlæg og responset i PoC. 
• De relevante simuleringsmodeller skal indeholde alle beskyttelsesfunktioner for for-
brugsanlægget og det samlede anlæg, som er relevante for det samlede anlægs dyna-
miske respons i PoC.  
• Systembrugeren er ansvarlig for at sikre retvisende dokumentation og verifikation af 
simuleringsmodellerne for forbrugsanlægget og det samplacerede anlægs respons i 
PoC. Dokumentation og verifikation skal inkludere angivelse af de anlæg i det sampla-
cerede anlæg, som bidrager til overholdelse af tekniske krav. 
• Systembrugeren har ansvar for at aggregere simuleringsmodellerne for det samlede 
anlæg i så stor udstrækning som muligt jf. ovenstående. Energinet kan godkende afvi-
gelser fra øvrige aggregeringskrav, hvis der kan argumenteres for, at en anden aggre-
gering af simuleringsmodellerne giver et væsentlig bedre repræsentation af det sam-
lede anlægs respons.  
 
5.22 PMU-måling  
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg påkræves der PMU-
måling til monitorering og verifikation af det samlede anlægs dynamiske respons. Hvis der er 
flere fysiske tilslutninger i samme PoC, skal der etableres PMU-måling i hver fysisk tilslutning. 
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.6.b.i 
• Energilageranlæg: Ingen eksisterende krav 
• Forbrugsanlæg: Ingen eksisterende krav. 
 
Produktionsanlæg: 
Produktionsanlæg, der samplaceres med energilager- og/eller forbrugsanlæg, skal monitoreres 
med PMU-måling. PMU-enheden/-enhederne skal etableres i de fysiske tilslutninger i PoC. Det 
påkræves kun, at der etableres en enkelt PMU-enhed i hver af de fysiske tilslutninger i PoC for 
det samlede anlæg, uagtet krav til PMU-måling for hhv. de samplacerede produktions-, for-
brugs- og/eller energilageranlæg. PMU-enhedens/-enhedernes tekniske specifikationer og da-
taudvekslingsformat besluttes af Energinet og fastsættes i nettilslutningsaftalen.  
 
Hvis det vurderes nødvendigt af Energinet, kan der yderligere blive påkrævet PMU-måling af 
produktionsanlægget placeret internt i det samlede anlæg. Dette vurderes af Energinet ifm. 
udarbejdelse af nettilslutningsaftalen. Den præcise placering af PMU-enhed/-enheder til moni-
torering af produktionsanlægget internt i det samlede anlæg udpeges af Energinet på basis af 
det samlede anlægs overordnede anlægsdesign. Placeringen afgøres i perioden mellem under-
skrift af nettilslutningsaftalen og EON.  
 
Energilageranlæg: 
Energilageranlæg, der samplaceres med produktions- og/eller forbrugsanlæg, skal monitoreres 
med PMU-måling. PMU-enheden/-enhederne skal etableres i de fysiske tilslutninger i PoC. Det 
påkræves kun, at der etableres en enkelt PMU-enhed i hver af de fysiske tilslutninger i PoC for 
det samlede anlæg, uagtet krav til PMU-måling for hhv. de samplacerede produktions-, for-
brugs- og/eller energilageranlæg. PMU-enhedens/-enhedernes tekniske specifikationer og da-
taudvekslingsformat besluttes af Energinet og fastsættes i nettilslutningsaftalen.  
 
33/39 
 
Dok.23/13192-1 Offentlig/Public 
Hvis det vurderes nødvendigt af Energinet, kan der yderligere blive påkrævet PMU-måling af 
energilageranlægget placeret internt i det samlede anlæg. Dette vurderes af Energinet ifm. ud-
arbejdelse af nettilslutningsaftalen. Den præcise placering af PMU-enhed/-enheder til monito-
rering af energilageranlægget internt i det samlede anlæg udpeges af Energinet på basis af det 
samlede anlægs overordnede anlægsdesign. Placeringen afgøres i perioden mellem underskrift 
af nettilslutningsaftalen og EON.  
 
Forbrugsanlæg: 
Forbrugsanlæg, der samplaceres med energilager- og/eller produktionsanlæg, skal monitoreres 
med PMU-måling. PMU-enheden/-enhederne skal etableres i de fysiske tilslutninger i PoC. Det 
påkræves kun, at der etableres en enkelt PMU-enhed i hver af de fysiske tilslutninger i PoC for 
det samlede anlæg, uagtet krav til PMU-måling for hhv. de samplacerede produktions-, for-
brugs- og/eller energilageranlæg. PMU-enhedens/-enhedernes tekniske specifikationer og da-
taudvekslingsformat besluttes af Energinet og fastsættes i nettilslutningsaftalen.  
 
Hvis det vurderes nødvendigt af Energinet, kan der yderligere blive påkrævet PMU-måling af 
forbrugsanlægget placeret internt i det samlede anlæg. Dette vurderes af Energinet ifm. udar-
bejdelse af nettilslutningsaftalen. Den præcise placering af PMU-enhed/-enheder til monitore-
ring af forbrugsanlægget internt i det samlede anlæg udpeges af Energinet på basis af  det sam-
lede anlægs overordnede anlægsdesign. Placeringen afgøres i perioden mellem underskrift af 
nettilslutningsaftalen og EON.  
 
5.23 Registrering af fejlhændelser (Transient Fault Recorder, TFR)  
Ved samplacering af produktions-, forbrugs- og/eller energilageranlæg påkræves fejlskrivere 
(også kaldet TFR) til eftervisning af krav og respons/karakteristika fra et givent anlæg. Hvis der 
er flere fysiske tilslutninger i samme PoC, skal der etableres TFR i hver fysisk tilslutning.  
Energinet specificerer krav til kvaliteten af måledata i nettilslutningsaftalen, herunder også ini-
tiering af TFR-logning. 
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.6.b.ii 
• Energilageranlæg: TF 3.3.1 §62 
• Forbrugsanlæg: NC DC art. 21.5. 
 
Produktionsanlæg: 
For produktionsanlæg, der samplaceres med energilager- og/eller forbrugsanlæg, skal der 
etableres TFR. TFR-enheden/-enhederne skal etableres i de fysiske tilslutninger i PoC. Det på-
kræves kun, at der etableres en enkelt TFR-enhed i hver af de fysiske tilslutninger i PoC for det 
samlede anlæg, uagtet krav til TFR for hhv. de samplacerede produktions-, forbrugs- og/eller 
energilageranlæg.  
 
Hvis Energinet vurderer det nødvendigt, kan der yderligere blive påkrævet etablering af TFR for 
produktionsanlægget placeret internt i det samlede anlæg. Dette vurderes af Energinet ifm. 
udarbejdelse af nettilslutningsaftalen. Den præcise placering af TFR-enhed/-enheder til moni-
torering af produktionsanlægget internt i det samlede anlæg udpeges af Energinet på basis af 
det samlede anlægs overordnede anlægsdesign. Placeringen afgøres i perioden mellem under-
skrift af nettilslutningsaftalen og EON.  
 
Energilageranlæg: 
34/39 
 
Dok.23/13192-1 Offentlig/Public 
For energilageranlæg, der samplaceres med produktions- og/eller forbrugsanlæg, skal der 
etableres TFR. TFR-enheden/-enhederne skal etableres i de fysiske tilslutninger i PoC. Det på-
kræves kun, at der etableres en enkelt TFR-enhed i hver af de fysiske tilslutninger i PoC for det 
samlede anlæg, uagtet krav til TFR for hhv. de samplacerede produktions-, forbrugs- og/eller 
energilageranlæg.  
 
Hvis Energinet vurderer det nødvendigt, kan der yderligere blive påkrævet etablering af TFR for 
energilageranlægget placeret internt i det samlede anlæg. Dette vurderes af Energinet ifm. ud-
arbejdelse af nettilslutningsaftalen. Den præcise placering af TFR-enhed/-enheder til monitore-
ring af energilageranlægget internt i det samlede anlæg udpeges af Energinet på basis af det 
samlede anlægs overordnede anlægsdesign. Placeringen afgøres i perioden mellem underskrift 
af nettilslutningsaftalen og EON.  
 
Forbrugsanlæg: 
For forbrugsanlæg, der samplaceres med energilager- og/eller produktionsanlæg, skal der 
etableres TFR. TFR-enheden/-enhederne skal etableres i de fysiske tilslutninger i PoC. Det på-
kræves kun, at der etableres en enkelt TFR-enhed i hver af de fysiske tilslutninger i PoC for det 
samlede anlæg, uagtet krav til TFR for hhv. de samplacerede produktions-, forbrugs- og/eller 
energilageranlæg.  
 
Hvis Energinet vurderer det nødvendigt, kan der yderligere blive påkrævet etablering af TFR for 
forbrugsanlægget placeret internt i det samlede anlæg. Dette vurderes af Energinet ifm. udar-
bejdelse af nettilslutningsaftalen. Den præcise placering af TFR-enhed/-enheder til monitore-
ring af forbrugsanlægget internt i det samlede anlæg udpeges af Energinet på basis af  det sam-
lede anlægs overordnede anlægsdesign. Placeringen afgøres i perioden mellem underskrift af 
nettilslutningsaftalen og EON.  
 
5.24 Produktions/Forbrugstelegraf 
Telegrafkoncept for samplacering bliver udarbejdet. 
 
5.25 Signalliste 
Signalliste for samplacering bliver udarbejdet. 
 
5.26 Køreplaner og tilsvarende målinger 
Målinger bliver fastsat i forbindelse med signaludveksling. 
 
5.27 Gensynkronisering 
Ved overplanting og/eller samplacering af produktions-, forbrugs- og/eller energilageranlæg 
skal de individuelle anlæg kunne følge de individuelle krav vedrørende gensynkronisering. 
Driftspunktet, som det samlede anlæg skal returnere til, afhænger af, om anlægget var i pro-
duktions- eller forbrugstilstand før bortkobling for at sikre ensartet respons fra anlæg tilsluttet 
det kollektive elsystem i samme driftstilstand.  
 
Eksisterende krav 
• Produktionsanlæg: NC RfG art. 15.5.c 
• Energilageranlæg: TF 3.3.1 §64 
• Forbrugsanlæg: NC DC art. 19.4.a-b. 
 
Produktionsanlæg: 
35/39 
 
Dok.23/13192-1 Offentlig/Public 
Produktionsanlægget skal have evnen til hurtig gensynkronisering inden for 15 minutter jf. kra-
vene i NC RfG art. 15.5.c. For overplantede produktionsanlæg skal minimum en delmængde af 
anlægget kunne gensynkronisere for at returnere til driftspunktet, som det samlede anlæg 
havde i PoC før bortkoblingen. Produktionsanlægget skal fortsat kunne overholde øvrige tekni-
ske krav efter gensynkronisering.  
 
Hvis produktionsanlæg samplaceres med energilager- og/eller forbrugsanlæg, skal produkti-
onsanlægget bidrage til, at det samlede anlæg kan opfylde krav om gensynkronisering. System-
brugeren har til ansvar at koordinere gensynkronisering for de samplacerede anlæg. Kravene 
differentieres afhængigt af, om det samlede anlæg var i produktions- eller forbrugstilstand 
forud for hændelsen: 
• Produktionstilstand: Det samlede anlæg skal returnere til driftspunktet, som det sam-
lede anlæg havde i PoC før bortkoblingen. Gensynkroniseringen skal ske inden for 15 
minutter.  
• Forbrugstilstand: Udvekslingen til det kollektive elsystem skal forblive nul, indtil  
KontrolCenter El giver tilladelse til ændringer.  
 
Energilageranlæg: 
Hvis energilageranlæg samplaceres med produktions- og/eller forbrugsanlæg, skal energilager-
anlægget bidrage til, at det samlede anlæg kan opfylde krav om gensynkronisering. Systembru-
geren har til ansvar at koordinere gensynkronisering for de samplacerede anlæg. Kravene diffe-
rentieres afhængigt af, om det samlede anlæg var i produktions- eller forbrugstilstand forud 
for hændelsen: 
• Produktionstilstand: Det samlede anlæg skal returnere til driftspunktet, som det sam-
lede anlæg havde i PoC før bortkoblingen. Gensynkroniseringen skal ske inden for 15 
minutter.  
• Forbrugstilstand: Udvekslingen til det kollektive elsystem skal forblive nul, indtil  
KontrolCenter El giver tilladelse til ændringer.  
 
Hvis energilageranlægget udelukkende samplaceres med et forbrugsanlæg, må det samlede 
anlæg ikke gensynkronisere, før KontrolCenter El giver tilladelse.  
 
Forbrugsanlæg: 
Hvis forbrugsanlæg samplaceres med produktions- og/eller energilageranlæg, skal forbrugsan-
lægget bidrage til at det samlede anlæg kan opfylde krav om gensynkronisering. Systembruge-
ren har til ansvar at koordinere gensynkronisering for de samplacerede anlæg. Kravene diffe-
rentieres afhængigt af, om det samlede anlæg var i produktions- eller forbrugstilstand forud 
for hændelsen: 
• Produktionstilstand: Det samlede anlæg skal returnere til driftspunktet, som det sam-
lede anlæg havde i PoC før bortkoblingen. Gensynkroniseringen skal ske inden for 15 
minutter.  
• Forbrugstilstand: Udvekslingen til det kollektive elsystem skal forblive nul, indtil  
KontrolCenter El giver tilladelse til ændringer.  
 
Hvis forbrugsanlægget udelukkende samplaceres med et energilageranlæg, må det samlede 
anlæg ikke gensynkronisere, før KontrolCenter El giver tilladelse.   
36/39 
 
Dok.23/13192-1 Offentlig/Public 
6. Bilag 
Følgende uddybelse af krav angiver de specifikke krav til systemværn (Afsnit 6.1) og PFAPR (Af-
snit 6.2), for produktions- og energilageranlæg, når det samlede anlæg er i forbrugstilstand, og 
for forbrugsanlæg, når det samlede anlæg er i produktionstilstand. Uddybelserne føjes til de 
tilhørende netregler for hver anlægstype ifm. implementering af kravene for overplanting og 
samplacering. 
 
6.1 Uddybelse af systemværnskrav 
6.1.1 Præcisering for produktions- og energilageranlæg angående forbrugstilstand 
Når det samlede anlæg er i forbrugstilstand, skal systemværnet som minimum overholde føl-
gende krav: 
a) Systemværnet skal kunne regulere det samlede anlægs aktive effektoptag til et af flere for-
uddefinerede reguleringstrin. 
b) Det samlede anlæg skal kunne indstilles med minimum fem forskellige konfigurérbare regu-
leringstrin. 
c) Reguleringen skal påbegyndes inden for 1 sekund og skal være fuldført inden for 10 sekun-
der fra modtagelse af ordre om regulering. 
Stk. 3. Reguleringstrinnene fastsættes af Energinet i koordination med systembrugeren  ved el-
ler efter indgåelse af nettilslutningsaftale og senest ved tildeling af ION. 
 
6.1.2 Præcisering for forbrugsanlæg angående produktionstilstand 
Når det samlede anlæg er i produktionstilstand, skal systemværnet på baggrund af en nedregu-
leringsordre meget hurtigt kunne regulere den aktive effekt leveret fra det samlede anlæg  til et 
eller flere foruddefinerede sætpunkter. Sætpunkterne fastlægges af elforsyningsvirksomheden 
ved idriftsættelsen. 
Anlægget skal have mulighed for minimum fem forskellige konfigurérbare reguleringstrin. 
Som standardværdier anbefales følgende reguleringstrin: 
1. Til 70 % af mærkeeffekt 
2. Til 50 % af mærkeeffekt 
3. Til 40 % af mærkeeffekt 
4. Til 25 % af mærkeeffekt 
5. Til 0 % af mærkeeffekt, dvs. anlægget er stoppet. 
Reguleringen skal påbegyndes inden for 1 sekund og skal være fuldført inden for 10 sekunder 
fra modtagelse af ordre om nedregulering. 
Hvis der til systemværnet beordres en opregulering, f.eks. fra trin 4 
(25 %) til 3 (40 %), accepteres det, at designmæssige grænser for anlæggets generatorer eller 
øvrige anlægsenheder kan give en forøget tid for fuldførelse af ordren. 
  
37/39 
 
Dok.23/13192-1 Offentlig/Public 
6.2 Uddybelse af PFAPR-krav 
6.2.1 Præcisering for produktions- og energilageranlæg angående forbrugstilstand  
Når det samlede anlæg er i forbrugstilstand, gælder følgende PFAPR-krav til indsvingningsfor-
løb, tid og nøjagtighed:  
 
 
 
hvor indsvingningsforløb og nøjagtighed skaleres med udvekslingskapacitet (P
nd3) frem for in-
stalleret effekt (PnD1). Der tillades en forskel i optag af aktiv effekt, før og efter hændelse, sva-
rende til 2 % Pnd3.  
 
6.2.2 Præcisering for forbrugsanlæg angående produktionstilstand 
Når det samlede anlæg er i produktionstilstand, gælder følgende PFAPR-krav til indsvingnings-
forløb, tid og nøjagtighed:  
 
Anlægget skal efter et indsvingningsforløb levere normal produktion senest 5 sekunder efter, 
at driftsforholdene i PoC er tilbage i området kontinuert drift. Nøjagtighed for en fuldført regu-
lering skal være i området +/- 5% af P
nG3, med forbehold for ændring i tilgængeligheden af pri-
mær energikilde. Effektreguleringen skal ske med en tilnærmelsesvis konstant gradient, hvor 
den aktive effekt under indsvingningsforløbet skal ligge inden for området defineret på F igur x, 
hvor:  
- T0 er tidspunktet, hvor driftsforholdene er tilbage i området kontinuert drift.  
- T1 er mellem 100 – 500 ms efter T0 og angiver tidspunktet, hvor anlægget forlader 
FRT-mode, når funktionen til forsinket exit af FRT-mode anvendes jf. krav nedenfor. 
Hvis funktionen ikke anvendes, er T1 = T0.  
- T2 er tidspunktet, hvor anlægget igen leverer normal produktion (kan være op til 5 se-
kunder efter T0).  

38/39 
 
Dok.23/13192-1 Offentlig/Public 
- Pref er udvekslingseffekt før fejl.  
- Pt+ og Pt- er hhv. Pref +/- 5% af PnG3. 
 
 
For produktionsanlæg af type D gælder derudover:  
Type D produktionsanlæg skal være i stand til at lave en langsommere og kontrolleret regule-
ring af aktiv effekt tilbage til normal produktion. Krav til langsom PFAPR er fastlagt i nedenstå-
ende og vist på Figur y: 
 
 

39/39 
 
Dok.23/13192-1 Offentlig/Public 
hvor indsvingningsforløbet og nøjagtighed skaleres med udvekslingskapacitet (PnG3) frem for 
installeret effekt (PnG1). Hvor Pref er udveksling til det kollektive elsystem før hændelsen. Pt+ 
og Pt- er hhv. Pref +/- 5% af PnG3. 
 
Tidspunktet, hvor den aktive effekt er tilbage til normal produktion, T2, skal kunne indstilles til 
mellem 5 og 20 sekunder med en opløsning på 1 sekund. Den aktive effekt må på intet tids-
punkt under reguleringen overstige Pt+, og skal yderligere ligge inden for den øvre og nedre 
tolerance som vist på Figur y.  
- Den øvre tolerance er linjen fra Plimit ved T0 og til Pt+ ved T2, hvor 
• Plimit er defineret som: Pfejl + 0,5 pu af PnG3.  
• Pfejl er den aktive effekt udveksling til det kollektive elsystem til tidspunkt T0. 
- Den nedre tolerance er linjen fra P = 0 ved T1 og til Pt- ved T2. 
 
Derudover gælder, at den maksimale gradient under reguleringen (mellem T0 og T2) ikke må 
overstige 25% af PnG3/s. 
 
Med henblik på at sikre, at anlæg ikke toggler ind og ud af FRT-mode, skal produktionsanlæg-
get kunne indstilles til at kunne blive i FRT-mode mellem 100 - 500 ms, efter spændingen i PoC 
er normaliseret i normaldriftsområdet. Funktionaliteten til at blive i FRT skal kunne deaktiveres, 
og de enkelte anlæg kunne behandles individuelt.  """,
    ),
    (
        "lv_mk_821_hibridatlauja",
        "Ministru kabineta 2023. gada 19. decembra noteikumi Nr. 821 -- hibridatlaujas (punkti 14, 15, 17)",
        "https://likumi.lv/ta/id/348679-noteikumi-par-atlaujam-jaunas-elektroenergijas-razosanas-iekartas-ieviesanai-vai-elektroenergijas-razosanas-jaudas-palielinasanai",
        "LV",
        "lv",
        # Retrieved via direct fetch of likumi.lv (Latvia's official
        # legislation portal). Content below is a content-faithful
        # structured summary of the actual fetched regulation page
        # (points 14/15/17 as the source page itself identifies them),
        # not a paraphrase from memory. Defines the "hibridatlauja"
        # (hybrid permit) -- Latvia's own dedicated dual-technology permit
        # category for combining generation types or generation+storage
        # at one connection point, replacing what would otherwise be
        # multiple separate permits.
        """\
Ministru kabineta 2023. gada 19. decembra noteikumi Nr. 821
"Noteikumi par atlaujam jaunas elektroenergijas razosanas iekartas un elektroenergijas
uzkratuves ieviesanai vai elektroenergijas razosanas jaudas palielinasanai"
Avots: likumi.lv (Latvijas oficialais tiesibu aktu portals)
URL: https://likumi.lv/ta/id/348679-noteikumi-par-atlaujam-jaunas-elektroenergijas-razosanas-iekartas-ieviesanai-vai-elektroenergijas-razosanas-jaudas-palielinasanai

PIEZIME PAR SATURA KVALITATI: Tiesi iegots no likumi.lv (Latvijas oficiala
likumdosanas portala); zemak izmantotais teksts ir satura zina precizs
strukturets kopsavilkums no reali iegutas lapas (nevis atmina rekonstruets),
konkretie punkti (14., 15., 17.) atzimeti tieksi, ka avota lapa tos identifice.

Regulas mervis: nosaka kartibu, kada tiek izsniegtas, parregistretas, anuletas
un pagarinatas atlaujas jaunu elektroenergijas razosanas iekartu ieviesanai vai
elektroenergijas razosanas jaudas palielinasanai, ka ari drosibas naudas apmeru,
maksasanas un atmaksasanas kartibu un atteikuma kriterijus.

14. punkts (hibridatlaujas piemerosanas joma): Ja elektroenergijas razotajs
plano ieviest vairakus elektroenergijas razosanas iekartu veidus vienā
pieslegumā punktā (piemeram, saules un veja elektrostacijas, vai veja
elektrostaciju un elektroenergijas uzkratuvi), tam ir jasanem atlauja
vairakiem elektroenergijas razosanas iekartu veidiem ("hibridatlauja").

15. punkts (jaudas proporciju prasibas): Hibridatlaujas turetajam jaievero
proporcijas starp dazadiem razosanas iekartu veidiem, ar pielaujamo novirzi
lidz 5%. Sis ierobezojums NEATTIECAS uz elektroenergijas uzkratuves iekartam --
tas var tikt ieviestas neatkarigi no proporciju prasibam.

17. punkts (piemerojamiba): Visas regulas prasibas, kas attiecas uz standarta
atlaujam, attiecas ari uz hibridatlauju turetajiem.

2025. gada 9. decembra grozijumi: hibridatlaujas turetajiem noteiktos
apstaklos vairs nav strikti jaievero jaudas proporcija, kas noradita
hibridatlauja; samazinats ari iesniedzamo dokumentu apjoms (vairs nav
nepieciesams droisbas naudas maksajuma uzdevums un Pielikums Nr.2 par
iekartas tehniskajiem raditajiem).

RELEVANCE HIBRIDPROJEKTIEM: Sis ir vienigais no visam 9 pardefinam valstim
apstiprinats, iestades apstiprinats DUALA ATLAUJU KATEGORIJAS mehanisms --
"hibridatlauja" pati par sevi ir specifisks, atseviskis atlaujas veids
kombinetiem BESS+veja/saules projektiem viena pieslegusa punkta, nevis
divas atsevisikas atlaujas, kas jaiegust paralel.""",
    ),
]


def _chunk(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end])
        start += CHUNK_CHARS - OVERLAP
    return chunks


def ingest(dry_run: bool = False) -> None:
    from sentence_transformers import SentenceTransformer
    import chromadb

    if not DB_DIR.exists():
        print(f"[ingest_hybridi_colocation] ERROR: {DB_DIR} missing. Run build_index.py first.")
        sys.exit(1)

    print(f"[ingest_hybridi_colocation] Connecting to ChromaDB: {DB_DIR}")
    model  = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col    = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    existing_ids: set[str] = set(col.get()["ids"])
    print(f"[ingest_hybridi_colocation] Existing chunks: {len(existing_ids)}")
    print()

    verified_today = time.strftime("%Y-%m-%d", time.gmtime())
    grand_new = 0

    for doc_id, source_label, url, country, lang, text in DOCS:
        chunks = _chunk(text)
        new_docs:  list[str]  = []
        new_ids:   list[str]  = []
        new_metas: list[dict] = []

        for i, chunk in enumerate(chunks):
            id_ = f"hybridi_colocation_inline__{doc_id}__{i}"
            if id_ in existing_ids:
                continue
            new_docs.append(chunk)
            new_ids.append(id_)
            meta = {
                "country":       country,
                "lang":          lang,
                "source":        doc_id,
                "source_type":   "manual",
                "last_verified": verified_today,
                "url":           url,
                # hanketyyppi_tag deliberately NOT set here -- see module
                # docstring; resolved exclusively via source_policy.py.
            }
            new_metas.append(meta)

        print(f"[{doc_id}] {source_label}")
        print(f"  Chunks: {len(chunks)} total, {len(new_docs)} new")

        if not new_docs:
            print("  -> Nothing new to add\n")
            continue

        if dry_run:
            print(f"  DRY-RUN: would add {len(new_docs)} chunks\n")
            grand_new += len(new_docs)
            continue

        for i in range(0, len(new_docs), BATCH):
            b = slice(i, i + BATCH)
            embs = model.encode(new_docs[b], show_progress_bar=False).tolist()
            col.add(
                documents=new_docs[b],
                embeddings=embs,
                ids=new_ids[b],
                metadatas=new_metas[b],
            )
        existing_ids.update(new_ids)
        grand_new += len(new_docs)
        print(f"  Added {len(new_docs)} chunks\n")

    print(f"{'-'*55}")
    print("Summary (hybridi co-location ingest):")
    print(f"  New chunks added: {grand_new}")
    print(f"  Total index size: {col.count()}")
    print(f"{'-'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index hybrid co-location Priority-2 sources into ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Show chunk counts, do not write")
    args = parser.parse_args()
    ingest(dry_run=args.dry_run)
