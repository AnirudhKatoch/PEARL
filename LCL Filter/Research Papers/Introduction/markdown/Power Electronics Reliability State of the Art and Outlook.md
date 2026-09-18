Aalborg Universitet
Power Electronics Reliability: State of the Art and Outlook
Wang, Huai; Blaabjerg, Frede
Published in:
IEEE Journal of Emerging and Selected Topics in Power Electronics
DOI (link to publication from Publisher):
10.1109/JESTPE.2020.3037161
Creative Commons License
CC BY 4.0
Publication date:
2021
Document Version
Publisher's PDF, also known as Version of record
Link to publication from Aalborg University
Citation for published version (APA):
Wang, H., & Blaabjerg, F. (2021). Power Electronics Reliability: State of the Art and Outlook. IEEE Journal of
Emerging and Selected Topics in Power Electronics, 9(6), 6476-6493. Article 9253639.
https://doi.org/10.1109/JESTPE.2020.3037161
General rights
Copyright and moral rights for the publications made accessible in the public portal are retained by the authors and/or other copyright owners
and it is a condition of accessing publications that users recognise and abide by the legal requirements associated with these rights.
- Users may download and print one copy of any publication from the public portal for the purpose of private study or research.
- You may not further distribute the material or use it for any profit-making activity or commercial gain
- You may freely distribute the URL identifying the publication in the public portal -
Take down policy
If you believe that this document breaches copyright please contact us at vbn@aub.aau.dk providing details, and we will remove access to
the work immediately and investigate your claim.
Downloaded from vbn.aau.dk on: June 15, 2026

6476 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
Power Electronics Reliability: State of
the Art and Outlook
Huai Wang , Senior Member, IEEE, and Frede Blaabjerg , Fellow, IEEE
Abstract—This article aims to provide an update of the Reliability of Power Electronics. In the same year, an edited
reliability aspects of research on power electronic components book on the reliability of power electronic converter systems
and hardware systems. It introduces the latest advances in the
was published by IET [3]. Research activities and outcomes
understanding of failure mechanisms, testing methods, accumu-
worldwide have increased significantly in the last decade.
lateddamagemodeling,andmission-profile-basedreliabilitypre-
diction. Component-level examples (e.g., Si IGBT modules, SiC Many of the results are reviewed in [4]–[13]. In [4], the
MOSFETs, GaN devices, capacitors, and magnetic components) research activities in the power cycling tests of power mod-
areusedforillustrationpurposesinadditiontosystem-levelstud- ules between 1994 and 2015 are summarized, which show
ies. The limitations and associated open questions are discussed
the varieties in packaging technologies, control strategies,
to identify future research opportunities in power electronics
failure mechanisms of interest, and failure analysis methods.
reliability.
In [5]–[7],themethodsforestimatingthejunctiontemperature
Index Terms—Accelerated degradation testing, condition
and health condition of power modules are reviewed. Fol-
monitoring, control, failure mechanism, physics-of-degradation,
lowing the adoption of wide bandgap devices in commercial
power electronics, reliability prediction.
applications, new packaging technologiesare discussed in [8]
and [9]. The chip- and packaging-level failure mechanisms,
I. INTRODUCTION
testing, and condition monitoring health precursors of SiC
EFFICIENCYandpowerdensityhavebeenattheheartof
devices are summarized in [10]. The failure mechanisms of
the power electronics community over the course of the
GaN devices are discussed in [11]. In addition to active
last five decades. The improvements in efficiency and power
components, capacitor reliability and condition monitoring
density are heavily driven by power semiconductor technolo-
methods are discussed in [12] and [13], respectively. These
gies, circuit topologies, and control methods. Fig. 1 shows
overviewarticlesprovideafoundationtounderstandthestate-
the historical development of power semiconductors, power
of-the-art research on power electronics reliability.
electronics, and reliability engineering. With the increasing
This article focuses on updating the power electronics
demandsofapplication-drivenresearch,reliabilityhasbecome
reliabilityaspectsofresearch,whicharelessaddressedin[1]–
aconsiderablepracticalchallengeinpowerelectronics,among
[13]. Most of the references that will be discussed in this
otherslisted inFig. 1.Reliabilityis animportantperformance
article are from the last five years. For a more comprehensive
factorthatisconsideredduringthedesign,manufacturing,and
list of references and discussions, please refer to [1]–[13].
field operation of power electronic converters. Over the past
It should be noted that this article limits its scopes in the
decade,thebusinessmodelintheindustryhasbeentransition-
following.
ing from product suppliers to holistic service providers. This
1) The component failure mechanisms are mainly on the
demands life-cycle-cost reduction and operation optimization
short circuit of SiC MOSFETs and GaN devices and
of power electronic converters through innovative design,
thewear-outofmagneticcomponents.Thefailuremech-
in-depth understanding of failure mechanisms and mission
anisms of Si IGBT modules and capacitors and the
profiles, and predictive maintenance.
degradation of SiC MOSFETs and GaN devices are
In 2013–2014, the authors reviewed the three aspects
discussed in detail in [10]–[12].
of research in power electronics reliability in [1] and [2].
2) Thediscussedstressorsarelimitedto theelectrothermal
In May 2015, the IEEE TRANSACTIONS ON POWER ELEC-
stresses and humidity. Other stressors (e.g., vibration
TRONICS published a Special Issue on Robust Design and
and environmental contaminants) and related failure
Manuscript received April 30, 2020; revised October 8, 2020; accepted mechanisms are not included.
October 20, 2020. Date of publication November 10, 2020; date of current 3) A comprehensive reliability prediction procedure for
version December 1, 2021. This work was supported in part by VIL-
power electronic converters is not included. Instead,
LUM FONDEN under the VILLUM Investigators Grant REPEPS (Award
00016591) and in part by Innovation Fund Denmark through the Advanced the references that give detailed discussions of this
Power Electronic Technology and Tools Project (APETT). Recommended topic are introduced. In addition, reliability prediction
forpublication by Associate Editor RonHui. (Corresponding author: Frede
methods are only briefly presented. For more details,
Blaabjerg.)
The authors are with the Department of Energy Technology, Aalborg please refer to [2].
University, 9220Aalborg,Denmark(e-mail:hwa@et.aau.dk; fbl@et.aau.dk). This article starts with a description of the relevance of
Color versions of one or more figures in this article are available at
component-levelfailuremechanismsinpowerelectronicappli-
https://doi.org/10.1109/JESTPE.2020.3037161.
Digital ObjectIdentifier 10.1109/JESTPE.2020.3037161 cations. Next, component degradation curves, lifetime model
Thisworkislicensed underaCreative CommonsAttribution 4.0License. Formoreinformation, seehttps://creativecommons.org/licenses/by/4.0/

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6477
Fig.1. Historical development ofpowersemiconductors, powerelectronics, andreliability engineering.
limitations, and time constraints of accelerated testing are operation. However, the problem here is that the dominant
discussed. Mission-profile-based reliability prediction and its failuremechanismsmayvarywiththedesigncriteriaandenvi-
challenges in thermal model simplification and accumulated ronmentalandoperationalconditionsofdifferentapplications.
damage modeling are also presented. Finally, a specific out-
look of power electronics reliability research is provided. A. Examples of Field Experiences
Anexampleoffieldexperiencesinphotovoltaic(PV)appli-
II. FAILURE IN POWERELECTRONIC cations is presented in [17], which is based on three data
COMPONENTS AND SYSTEMS reports (primarily for central inverters). Report 1 includes
In reliability engineering, failure is typically classified into more than 3500 service tickets (i.e., abnormal states of PV
sudden failure and degraded failure. Degradation in time may systems) that were requested during 2010–2012 with a total
ultimately cause a sudden failure if no corrective action is operationof2800inverteryears[18].Report2isfromanolder
taken.Suddenfailureiscausedbydesigndefects,manufactur- set of PV systems monitored during 2008–2010 with a total
ing issues, single-event effects, overstress, or misuse. In con- of 1650 inverter years. Report 3 is based on about 400 ser-
trast, degraded failure is caused by long-term degradation. vice tickets. According to [17, Fig. 1], inverters account for
For a power electronic system, failure can come from both 43%–70% of PV plant service requests. Inverter component
hardware and software due to intrinsic and extrinsic factors. failurebreakdownbased on the three data sourcesis analyzed
This article focuses on hardware failures; consequently, soft- in [17, Fig. 3]. Controlsoftware failure stands out in all three
ware reliability [14] and human reliability analysis [15] are reports. This specific failure area may be the consequence of
not included. Nevertheless, they are essential to a complete hardware malfunction, besides software or firmware issues.
system-level reliability design and analysis. The inverters may resume normal operation upon a manual
Theoretically,multiplefailuremechanismsexistforasingle restart [18]. Although the importance of the widely studied
powerelectroniccomponent[10]–[12],[16].Attheapplication hardwarecomponents(e.g.,IGBTsand capacitors)remainsin
level, it might not be realistic to consider all of the reported termsoftheirfailurepercentageasasingletypeofcomponent,
failure mechanisms. A more effective method is to address other failure areas, including control card PCB boards, fans,
a few dominant failure mechanisms that are relevant in field and fuses, also rank high. However, in-depth failure analyses

6478 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
are unavailable, and the causes of failure and the dominant capability. Nevertheless, the compromised short-circuit with-
stressors are still open to question. In particular, it would stand time with the reducedSiC die size remainsa challenge.
be interesting to study humidity-related failure mechanisms Much effort has been made in the short-circuit
due to the condensation effect of frequent cold starts in the testing [27]–[34] of SiC MOSFETs and its protection
intermittent operation of PV inverters. in power converter design [34]–[38]. In [27], comparative
In[19],anexampleofwindpowerconverterfailureanalysis testing of two types of 1.2-kV/24-A SiC MOSFETs with
is given.Thephase modulesare one areaof failure,including a Si counterpart is given. Only a gradual reduction of
IGBTmodules,gatedriverboards,dc-linkcapacitors,andbus- gate–source voltage is observed from the testing samples
bars. The database used for this analysis includes 2734 wind under short-circuit operation. This is caused by the gate
turbines that were commissioned between 1997 and 2015, leakage current triggered by excessive power dissipation,
operating in 23 countries from 11 suppliers. As shown in resulting in lower gate-oxide reliability due to a thinner
[19, Fig. 3], most wind turbines are either in the early or gate thickness. In [28], 1.2-kV/36-A SiC devices are tested
middle operation phase when considering a typical service with short-pulse high voltage and long-pulse low voltage.
life of 25 years. According to [19, Fig. 5], except for other Two different failure mechanisms are observed: first, thermal
unknown failure sources, phase modules are responsible for runaway due to high OFF-state drain leakage current; second,
the largest fraction of power converter failures (i.e., 22%), gate–source shorting due to breaking up of top layers. The
which is followed by the cooling system, control board, and observed failure mechanisms are consistent with [29] in
the main circuit breaker. On average, 0.16 phase-module the case studies of 1.2-kV/180-A and 1.2-kV/300-A SiC
failure per year per turbine is observed for these turbines. MOSFETmodules.Under600-Vtesting,thethermalrunaway
The detailed results question the widespread assumption that becomes dominant according to the switching waveforms
thermal-related wear-out is the dominant failure mechanism shown in Fig. 2(a). The saturation current reaches about
of these phase modules. Instead, it indicates that there is a 18 times the nominal value. It then drops quickly before the
correlation between the failure rate and the absolute humidity switch is turned off, which indicates a significant increase in
level with seasonal change, which suggests that humidity and the chip temperature. A delayed failure occurs at 2 μs after
condensationmaybethemainstressorscausingfailure.These the turn-off due to excessive drain leakage current at high
observations are consistent with the findings in [20]. temperature. When the testing voltage increases to 800 V,
These examples represent two large databases for gate–sourceshorting is observed.This is due to the increased
converter-specific failure analyses, which are rarely made gate leakage current triggered by high local temperatures
publicly available in other power electronic applications. close to the gate oxide. The corresponding switching
Both findings suggest the need for further investigation of waveforms are shown in Fig. 2(b). A safe operation area is
humidity-related failure mechanisms for power electronic proposed in terms of gate–source voltage and short-circuit
components [21], [22]. Nevertheless, they do not necessarily current [29]. A case study of a 10-kV/10-A SiC MOSFET
implythatthethermal-relatedwear-outfailuremechanismsare tested under 6 kV is presented in [30]. The testing results do
irrelevant.Mostof the analyzedwind turbinesin [19]are still not show signs of measurable thermally generated currents
in the early-to-middle phase of their service life. A proper or degradation in the gate structure. The impact of thermal
design and manufacturing process aims to prevent or, to a stress on short-circuit capability is studied in [31] and [32]
largeextent,reducewear-outfailure.Thermal-relatedwear-out based on the testing of 1.2-kVSiC MOSFETs underdifferent
prediction can inform decisions in the design and operation case temperatures. This shows that the short-circuit critical
phases to prevent the associated failure from becoming dom- energy and the withstand time slightly decrease with the case
inant during the service life. temperature. Moreover, the relationship between the number
of repetitive short circuits to failure and the case temperature
is presented [32]. The gate–source voltage drop from the
initial value of 15 V at the time of 2 μs in short-circuit
B. Selected Failure Mechanisms
operation is used as a failure criterion. This shows that with
1) Short Circuit and Overstress Failure of SiC MOSFETs: the increase in case temperature, the number of repetitive
ThereliabilityperformanceofSiCMOSFETsisofsignificant shortcircuitsto failureincreases. It is becausethatthe energy
consideration in power electronic applications. Extrinsic fail- dissipated during the short circuit operation decreases with
uresduetodefectivityorhighprocessvariabilityandintrinsic the increase of temperature due to the negative temperature
failures due to material degradation are both relevant [23]. coefficient of the MOSFETs. The impact of degradation
On the one hand, field experiences are limited by the early on the short-circuit withstanding capability of a 1.2 kV
stage of technology adoption. On the other hand, unique SiC MOSFET module has been investigated [33] based on
reliability issues compared with Si devices give a lower comparisons between new samples and aged samples from
cycling capability with conventional packaging and reduced power cycling test. This study shows that the gate-oxide
short-circuitwithstandtime[24].Thepowercyclingcapability degradationhas a more significant impact on the short-circuit
of SiC MOSFETs has been reported to be lower than IGBTs withstand time than the packaging-leveldegradation.
with the same current rating and package technology by These studies give us a better understanding of the
testing [25]and by simulation [26].New packagingtechnolo- short-circuitfailuremechanismsofSiCMOSFETsandmutual
gies[9]arelikelytoovercomethelimitationinpowercycling effectswithchip-levelandpackaging-leveldegradation.Itcan

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6479
|     |     |     |     |     |     | failures                             | of GaN        | devices      | need        | to       | be considered              |                   | for          | power   |
| --- | --- | --- | --- | --- | --- | ------------------------------------ | ------------- | ------------ | ----------- | -------- | -------------------------- | ----------------- | ------------ | ------- |
|     |     |     |     |     |     | electronic                           | applications. |              | The         | level of | extrinsic                  | failures          |              | of GaN  |
|     |     |     |     |     |     | devices                              | due to        | material     | defectivity |          | still                      | has not           | reached      | the     |
|     |     |     |     |     |     | level of                             | the Si        | counterparts |             | [40].    | The internal               |                   | GaN-specific |         |
|     |     |     |     |     |     | degradationmechanismsduringthe       |               |              |             |          | OFF-state,switching-state, |                   |              |         |
|     |     |     |     |     |     | and ON-state                         | are           | presented    |             | in [11]  | and [41].                  | In                | [41],        | a power |
|     |     |     |     |     |     | cycling                              | test is       | designed     | to          | exclude  | thermomechanical           |                   |              | effects |
|     |     |     |     |     |     | bysettingtemperatureswingslessthan25 |               |              |             |          |                            | ◦ C.Thisworkstud- |              |         |
iesthethresholdvoltageshiftanddynamicON-stateresistance
|     |     |     |     |     |     | change           | due to | the trapping  |     | effect. | Thermomechanical-related |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------------- | ------ | ------------- | --- | ------- | ------------------------ | --- | --- | --- |
|     |     |     |     |     |     | powercyclingtest |        | isperformedin |     |         | [42]–[45]withtemperature |     |     |     |
◦
|     |     |     |     |     |     | swings        | of 100           | C or         | above.     | The        | drain-to-source |         | OFF-state      |         |
| --- | --- | --- | --- | --- | --- | ------------- | ---------------- | ------------ | ---------- | ---------- | --------------- | ------- | -------------- | ------- |
|     |     |     |     |     |     | leakage       | current          | (IDSS)       | failure    | is         | reported        | in      | [42] and       | [43].   |
|     |     |     |     |     |     | The failure   | mechanism        |              | is         | identified | in              | [44]    | by postfailure |         |
|     |     |     |     |     |     | analysis      | tools            | and detailed |            | FEM        | simulations.    |         | This           | reveals |
|     |     |     |     |     |     | that the      | thermomechanical |              |            | stresses   | induced         | by      | an acceptable  |         |
|     |     |     |     |     |     | temperature   | range            | can          | cause      | multilayer |                 | cracks  | in the         | die of  |
|     |     |     |     |     |     | the GaN       | device.          |              |            |            |                 |         |                |         |
|     |     |     |     |     |     | Short-circuit |                  | failure      | mechanisms |            | are             | studied | in [46]–[49].  |         |
Thedrain-to-sourcevoltagedistinguishestwofailuremodesof
GaNdevices.Acasestudyona650-Vp-GaNisgivenin[46]
|     |     |     |     |     |     | and [47]. | The | short-circuit |     | withstand | time | is in | the | range of |
| --- | --- | --- | --- | --- | --- | --------- | --- | ------------- | --- | --------- | ---- | ----- | --- | -------- |
fewμsormorewhenthetestingvoltageisbelow350V.This
|     |     |     |     |     |     | reduces | to a few | hundreds |     | of ns | when | the voltage |     | is above |
| --- | --- | --- | --- | --- | --- | ------- | -------- | -------- | --- | ----- | ---- | ----------- | --- | -------- |
350V.Thefirstfailuremodeisduetothetemperatureincrease
|     |     |     |     |     |     | in the entire | device                                 | (i.e.,  | more    | than        | 1000       | K), causing     |             | damage |
| --- | --- | --- | --- | --- | --- | ------------- | -------------------------------------- | ------- | ------- | ----------- | ---------- | --------------- | ----------- | ------ |
|     |     |     |     |     |     | to the        | surface                                | metal   | [48],   | [49].       | The second |                 | failure     | mode   |
|     |     |     |     |     |     | is related    | to local                               | thermal |         | destruction |            | in a relatively |             | small  |
|     |     |     |     |     |     | regionof      | the device[46],[48],[49].Especially,it |         |         |             |            |                 | is observed |        |
|     |     |     |     |     |     | that the      | right-hand                             |         | side of | the gate    | field      | plate           | is the      | most   |
|     |     |     |     |     |     | sensitive     | part of                                | the     | testing | samples     | [49].      | It can          | be noted    | that   |
sub-μs
Fig.2. Short-circuittestingofa1.2-kV/300-ASiCMOSFETmoduleunder destructive short circuits could happen within under
◦
different dc voltages and case temperature of 25 C. ID: drain current; moderate-voltagestresses.Therefore,thedesignofanultrafast
VGS:gate–sourcevoltage;andVDS:drain–sourcevoltage.Gatedriversupply
+20 V/−6V  overcurrentprotection solution [50] is even more challenging
|     | with an external | gate resistance | of 5 | [29]. (a) | Short-circuit |     |     |     |     |     |     |     |     |     |
| --- | ---------------- | --------------- | ---- | --------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thermalrunawayfailureoccursat600-Vtestingvoltage.(b)Short-circuitgate than that for SiC MOSFETs. The drain-to-source voltage
breakdown failure occursat800-Vtesting voltage. stress is a vital factor determining short-circuitcapability and
dominantfailuremechanismsofbothSiCMOSFETsandGaN
| be noted | that varieties | still exist | in testing | conditions |     | in devices. |     |     |     |     |     |     |     |     |
| -------- | -------------- | ----------- | ---------- | ---------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
terms of the dc supply voltage, gate driver voltage, and 3) Reliability ofMagnetic Componentsin Power Electronic
case temperature due to a lack of standardization. Moreover, Applications: The transformersand inductorsthat are used in
thestudyofhigh-voltageorhigh-currentdevicesisstilllimited powerelectronicconvertersaregenerallyrecognizedashaving
due to the scarcity of the samples and higher demands on fewer failure issues than capacitors and power semiconductor
the testing facilities. With the increasing commercial appli- switches. Nevertheless, design-to-limit could become a stan-
cations of SiC devices, it is expected that more studies on dardpracticetomeetthedemandforcostreductionandpower
short circuits will be needed. At the application level, fast density increase. In these design scenarios, thermal-related
and robust short-circuit protection [34]–[38] is essential to failureandinsulation-relatedwear-outbecomerelevantwithin
utilize the promisingpropertiesof SiC power deviceswithout theservicelifeofpowerelectronicconverters.Moreover,insu-
compromising the device-level design. lation degradationor sudden breakdownis further accelerated
Overvoltage and overcurrent caused by parasitics and sus- by thermal stress, moisture, and dust levels.
tained oscillations are also important issues related to the The reliability studies of medium- to high–frequencymag-
reliability and EMI [39] in SiC applications. An excessive netic components for power electronic applications are rarely
| EMI may | lead to a | severe consequence | of  | failing | to fulfill |            |     |       |         |               |     |     |          |      |
| ------- | --------- | ------------------ | --- | ------- | ---------- | ---------- | --- | ----- | ------- | ------------- | --- | --- | -------- | ---- |
|         |           |                    |     |         |            | available. | The | first | studies | are presented |     | in  | [51] and | [52] |
the relevant standards and regulations. Advanced packaging on the degradation testing and lifetime modeling of pla-
technologies[9]andconverter-levelreductionofparasiticsand nar transformers. Two groups of 12 samples are tested
|     |     |     |     |     |     |     | ◦   |     |     | ◦   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
couplings between the power loop and gate driver loop can under 200 C and 180 C, respectively, for 3500 and
help address this issue [39]. 2500 h. The primary-side inductance reduces by 15%–40%.
2) Wear-OutandShort-CircuitFailureMechanismsofGaN Two of the possible failure mechanisms are core material
Devices: Like SiC MOSFETs, both the extrinsic and intrinsic degradation and expansion of the glue between two cores.

6480 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
Further investigations are needed to confirm the relevant
failure mechanisms.
Understandingthe relevantfailuremodesandfailure mech-
anisms is the first step to prioritizing resources for reliabil-
ity testing, lifetime and reliability prediction, and condition
monitoring.
III. DEGRADATION AND LIFETIMEOF POWER
ELECTRONICCOMPONENTS
A. Component Degradation Curve and End-of-Life Criteria
The degradation of power electronic components typically
occurs at the material or interconnection level, which needs
advanced physics analysis tools [44] to identify the loca-
tionandstructurechange.Alternatively,certainelectrothermal Fig.3. Threepossiblestages ofacomponent degradation curve. y isthe
parameter shift of the health precursor y, and y0 is the initial value of the
parameters can be used as health precursors. The values of
parameter x beforetestingorinuse.
them change with the degradation level of materials and
interconnects. For example, ON-state saturation voltage and to the time when the component destruct limit reaches or a
thermal resistance between junction-to-case are widely used system-level failure occurs. Therefore, the change rate of y
health precursors for IGBT modules, and they are applied for plays an essential role in defining the EOL but not neces-
both accelerated degradation testing [53] and condition mon- sarily the absolute value of y. This implies that component-
itoring [6]. Capacitance, equivalent series resistance (ESR), and system-level failures would occur soon if no action is
dissipation factor, and insulation resistance are usually used taken at the EOL. Meanwhile, the EOL can also be decided
for capacitors [12], [13]. accordingto the componentparameterconstraintsconsidering
Fig. 3 shows a generic degradation curve of a precursor system-level requirements.
corresponding to a specific failure mechanism. y is the value
of the health precursor and y is its parameter shift. The
B. (Percentile) Lifetime Definition
y-axis shows the absolute percentage change of y to its
Lifetimeis a widely usedreliability metricfora population
initial value y . The degradation curve has three possible
0
ofitemsdefinedataspecificreliabilitylevel.Foranindividual
distinctive stages: I, II, and III. These stages correspond to
item, time-to-failure is used corresponding to when the EOL
thetimeintervalsinwhichthehealthprecursorkeepsconstant,
isreached.Fig.4showsaWeibullplot[54]ofanunreliability
increasesordecreaseslinearly,andincreasesordecreaseswith
curve. The curve is plotted based on the time-to-failure data
an accelerated pace.
of a limited number of samples. The confidence intervals
It should be noted that Fig. 3 is given for illustration
illustrate the uncertainties. The solid line and dashed line
purposes only. Not all of the three stages necessarily appear
represent50% and 95% confidence levels, respectively. B is
for a specific precursor. A typical degradation curve is a X
used to define the percentile lifetime corresponding to when
combinationofoneormoreofthethreestages.Theslopeand
there is X% of accumulated failure. It can be noted that
percentageof each period vary for differentfailure precursors
the obtained lifetime varies with X% and confidence level.
and stress conditions.
B0.1_95%implies95%probabilitythat0.1%ofthepopulation
The time-to-failure of an individual component due to
of items will fail until the time. In other words, statistically,
wear-out is determined by one or more defined end-
there is a 5% risk that more than 0.1% of the items fail until
of-life (EOL) criteria that are chosen by considering the
the time. B10_50% implies 50% probability that 10% of the
component-leveldestructlimit,system-levelspecification,and
items fail until the time. Moreover, the plots shown in Fig. 4
a certain margin; as shown in Fig. 3. In practice, this is
are for specific stress conditions. Therefore, a comprehensive
achieved in terms of the percentage of change of y to its
lifetime definition should include at least the following four
initial value y . It is reported in [4] that a 5% increase in
0 aspects:
ON-state saturation voltage or a 20% increase in junction-
to-case thermal resistance is used as EOL in most of the 1) environmentaland operational conditions;
analyzed 70 publications on power cycling. This is consistent 2) EOL used for determining the time-to-failure;
with the recommendations by [53]. Nevertheless, 13% of the 3) correspondingpercentage of accumulated failure (X%);
publicationsusea20%increaseintheON-statesaturationvolt- 4) confidence level.
ageasEOL.Forelectrolyticcapacitors,20%capacitancedrop Nevertheless, such comprehensive information is rarely pro-
andtwotothreetimesincreaseinESRordissipationfactorare vided in the literature when lifetime is stated, which means
commonly used [12]. The EOL in terms of capacitance drop that we do not know the implication for failure and the
for film capacitors is usually within the range of 2%–10%. associated risk. If any aspect of the information is missed,
The varieties in EOL in terms of the percentage of change then the lifetime comparisonwould be in question.Moreover,
dependmainlyonwhenthedegradationacceleratesandenters the B lifetime with a certain confidence level is a single
X
StageIII,asshowninFig.3.Thisdoesnotusuallycorrespond point in the unreliability curve. The same B lifetime for
X

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6481
|     |     |     |     |     |     |     |     | a humidity | derating | factor, |     | as shown | in  | (4) is added | to (2) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ------- | --- | -------- | --- | ------------ | ------ |
in[55].In[56],threeotherfactorsrelatedtothepowermodule
|     |     |     |     |     |     |     |     | design parameters  |           | are          | included.                        |              |                 |                    |              |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --------- | ------------ | -------------------------------- | ------------ | --------------- | ------------------ | ------------ |
|     |     |     |     |     |     |     |     | Physics-based      |           | lifetime     | models                           | are          | also            | available          | for power    |
|     |     |     |     |     |     |     |     | modules,as         | givenin   | [3,          | Ch.5].Thesemodelsprovideinsights |              |                 |                    |              |
|     |     |     |     |     |     |     |     | into improving     |           | component    | design                           |              | through         | physics            | analyses.    |
|     |     |     |     |     |     |     |     | They can           | verify    | the assumed  |                                  | failure      | mechanisms      |                    | used for the |
|     |     |     |     |     |     |     |     | empirical          | models.   | At the       | application                      |              | level,          | empirical          | lifetime     |
|     |     |     |     |     |     |     |     | models, as         | given     | by (1)–(4),  |                                  | are commonly |                 | used               | due to their |
|     |     |     |     |     |     |     |     | simple form        | and       | direct       | correlation                      |              | with            | the electrothermal |              |
|     |     |     |     |     |     |     |     | stresses and       | other     | stressors.   |                                  | They         | also            | give insights      | into         |
|     |     |     |     |     |     |     |     | lifetime extension |           | by           | reducing                         | the          | stresses        | through            | power        |
|     |     |     |     |     |     |     |     | converter          | design    | and control. |                                  |              |                 |                    |              |
|     |     |     |     |     |     |     |     | The following      |           | limitations  |                                  | of existing  | lifetime        |                    | models are   |
|     |     |     |     |     |     |     |     | identified         | according | to           | the literature                   |              | and experience. |                    |              |
Fig.4. Illustrationofpercentilelifetimedefinitionunderdifferentconfidence
|     |     |     |     |     |     |     |     | 1) Limited | information |     | is  | available | for | EOL, | B lifetime |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | --- | --------- | --- | ---- | ---------- |
levelsforaspecificwear-outfailuremechanismbasedontime-to-failuredata X
inaWeibullplot. definition,andconfidencelevelwhenalifetimemodelis
|                    |     |              |     |                           |     |     |     | presented. |          | In [55] | and       | [57], comprehensive |       |        | degradation |
| ------------------ | --- | ------------ | --- | ------------------------- | --- | --- | --- | ---------- | -------- | ------- | --------- | ------------------- | ----- | ------ | ----------- |
|                    |     |              |     |                           |     |     |     | data       | analyses | are     | presented |                     | based | on the | testing of  |
| two differentitems |     | doesnotimply |     | equalreliabilityduringthe |     |     |     |            |          |         |           |                     |       |        |             |
capacitorsandpowermodules,respectively.Thisreveals
servicelifebecausetheslopeoftheirdegradationcurvescould
|                |        |             |          |           |              |          |     | that              | the obtained |       | B   | values                       | vary    | significantly  | with      |
| -------------- | ------ | ----------- | -------- | --------- | ------------ | -------- | --- | ----------------- | ------------ | ----- | --- | ---------------------------- | ------- | -------------- | --------- |
| be different.  |        |             |          |           |              |          |     |                   |              |       | X   |                              |         |                |           |
|                |        |             |          |           |              |          |     | lifetime          | definitions  |       | and | confidence                   | levels. |                |           |
|                |        |             |          |           |              |          |     | 2) Limited        | information  |       | is  | available                    | for     | the associated | fail-     |
| C. Limitations |        | of Existing | Lifetime | Models    |              |          |     |                   |              |       |     |                              |         |                |           |
|                |        |             |          |           |              |          |     | uremechanismwhena |              |       |     | lifetimemodelispresented.The |         |                |           |
| The most       | widely | considered  |          | stressors | in empirical | lifetime |     |                   |              |       |     |                              |         |                |           |
|                |        |             |          |           |              |          |     | power             | cycling      | tests | are | designed                     |         | to trigger     | different |
modelsforpowerelectroniccomponentsaretemperature,tem- failure mechanisms in bond wires and solder joints
| perature | swing, | and voltage. |     | The capacitor | lifetime | model | is  |     |       |         |     |      |           |     |             |
| -------- | ------ | ------------ | --- | ------------- | -------- | ----- | --- | --- | ----- | ------- | --- | ---- | --------- | --- | ----------- |
|          |        |              |     |               |          |       |     | of  | power | modules | in  | [58] | and [59]. | For | capacitors, |
givenby(1)anda simplifiedversionby(2)[11].Thelifetime lifetime models for different failure mechanisms [11]
model for the packaging related wear-out of power modules are unavailable, to the best of our knowledge.
| is given | by (3). | Relative |     | humidity | (RH) is an | additional |     |        |            |     |        |        |      |          |           |
| -------- | ------- | -------- | --- | -------- | ---------- | ---------- | --- | ------ | ---------- | --- | ------ | ------ | ---- | -------- | --------- |
|          |         |          |     |          |            |            |     | 3) The | applicable |     | stress | ranges | of a | lifetime | model are |
stressor, as described by [55]. The acceleration factor AF RH rarely investigated in reliability prediction. Fig. 5 illus-
| is presented | by  | (4) |     |     |     |     |     |        |     |            |     |         |             |     |            |
| ------------ | --- | --- | --- | --- | --- | --- | --- | ------ | --- | ---------- | --- | ------- | ----------- | --- | ---------- |
|              |     |     |     |     |     |     |     | trates | the | boundaries |     | for the | application | of  | a specific |
(cid:2) (cid:3)
|     |     |     | −n1 |     |     |     |     | lifetime                                          | model. |     | S and | S   | represent | the | lower and |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | ------ | --- | ----- | --- | --------- | --- | --------- |
|     |     | V   |     |     |     |     |     |                                                   |        |     | L     | H   |           |     |           |
| L = | L × |     |     |     |     |     |     | higherboundariesoftheconsideredstressor.Theunder- |        |     |       |     |           |     |           |
ref
|     | (cid:4)(cid:2) | V ref (cid:3)(cid:2) |     | (cid:3)(cid:5) |     |     |     |       |            |     |     |            |       |     |            |
| --- | -------------- | -------------------- | --- | -------------- | --- | --- | --- | ----- | ---------- | --- | --- | ---------- | ----- | --- | ---------- |
|     |                |                      |     |                |     |     |     | lying | assumption |     | of  | the models | given | in  | (1)–(3) is |
|     |                | E                    | 1   | 1              |     |     |     |       |            |     |     |            |       |     |            |
×exp a − ×other that the dominant failure mechanism remains the same
|     |     |     |     |     | factors |     | (1) |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | K   | T   | T   |         |     |     |     |     |     |     |     | L   | L   |     |
(cid:2) B(cid:3) ref within the predicted lifetime. 1 and 2 in Fig. 5
−n1
V Tre f −T s h o w t h e i n t e r v al w h e n th e li fe t im e m o d e l is a pp li c ab l e .
| L = | L × |     | ×   | 2 n | ×other factors |     | (2) |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ref 2 C o n se q u e n t l y , if th e p r ed ic te d li f eti m e i s lo n ge r th a n L ,
|     |          | V ref |      | (cid:2) (cid:3) |             |     |     |                                                     |     |     |     |     |     |     | 2   |
| --- | -------- | ----- | ---- | --------------- | ----------- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|     |          |       |      | E               |             |     |     | theresultsshouldbeinterpretedcautiously.Forexample, |     |     |     |     |     |     |     |
| =   | A×(T)−β |       | ×exp | a               | ×t β ×other |     |     |                                                     |     |     |     |     |     |     |     |
N 1 o n 2 factors in aluminum electrolytic capacitors, the lifetime model
K T
|     |     |     |     | B   |     |     |     | (2) | is applied. |     | In [60], | it is | stated | that | the lifetime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | ----- | ------ | ---- | ------------ |
(3)
=
|     |     |     |     |         |         |     |     | model                                           | is  | applicable | for | up  | to 15 | years | (i.e., L |
| --- | --- | --- | --- | ------- | ------- | --- | --- | ----------------------------------------------- | --- | ---------- | --- | --- | ----- | ----- | -------- |
|     |     |     |     | (cid:2) | (cid:3) |     |     |                                                 |     |            |     |     |       |       | 2        |
|     |     |     |     |         | −n3     |     |     | 15years).Onereasonforthisisthatanewfailuremech- |     |            |     |     |       |       |          |
RH
=
AF RH (4) anism (e.g., sealing rubber degradation) may become
RH
|     |     |     |     | ref |     |     |     | dominant |     | after | 15 years | of  | operation, | instead | of the |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | ----- | -------- | --- | ---------- | ------- | ------ |
where L isthepredictedlifetimeatthestresslevelofinterest. considered electrolyte evaporation and electrochemical
N isthepredictedcycle-to-failureatthestresslevelofinterest.
reaction.Italsoindicatesthattheapplicablerangeofthe
T
V is the voltage stress, T is the temperature, is the voltage stress in (2) is at the rated voltage or above for
| temperature | variation, |     | t is | the heating | time of | the thermal |     |                                                     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | ---- | ----------- | ------- | ----------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|             |            |     | on   |             |         |             |     | small-sizecapacitors.Thevoltagestresswithintherated |     |     |     |     |     |     |     |
cycle, E a istheactivationenergy, K B isBoltzmann’sconstant onehasanegligibleimpactontheirdegradation.In[61],
−5
(8.62×10 eV/K), and RH is the relative humidity. L is the applicable ranges of different factors are given for
ref
| the lifetime | at  | the referenced |     | operation | conditionsof | V   | and |     |      |       |        |          |       |       |        |
| ------------ | --- | -------------- | --- | --------- | ------------ | --- | --- | --- | ---- | ----- | ------ | -------- | ----- | ----- | ------ |
|              |     |                |     |           |              |     | ref | the | IGBT | power | module | lifetime | model | based | on the |
T ref . RH ref is the referenced RH for the acceleration factor testingconditions.Nevertheless,inmanyotherstudiesin
| AF . The | parameters |     | L   | , A, E | , n , n , n , | β , and | β   |     |     |     |     |     |     |     |     |
| -------- | ---------- | --- | --- | ------ | ------------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
RH ref a 1 2 3 1 2 theliterature,suchinformationisunavailable.Therefore,
are constants obtained based on time-to-failure data. Other further investigations of the specific boundaries shown
factors are not widely considered or unknown. For example, in Fig. 5 are suggested.

6482 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
|     |     |     |     | respectively. |                  | This corresponds |              | to the           | dominant       | failure          | mech-    |
| --- | --- | --- | --- | ------------- | ---------------- | ---------------- | ------------ | ---------------- | -------------- | ---------------- | -------- |
|     |     |     |     | anism         | of chip          | solder           | fatigue      | and bond         | wire wear-out, |                  | respec-  |
|     |     |     |     | tively.       | Humidity-related |                  | failure      | mechanisms       |                | are investigated |          |
|     |     |     |     | in [55],      | [64],            | and [65]         | for          | SiC MOSFET       | modules,       |                  | Si IGBT  |
|     |     |     |     | modules,      | and              | film capacitors, |              | respectively.    |                | As discussed     | in       |
|     |     |     |     | Section       | II               | on the field     | experiences, | humidity-related |                |                  | failures |
arelikelytobedominantoverthosecausedbythermalstresses
|     |     |     |     | for | the surveyed | systems |     | in the early-to-middle |     |     | stage of |
| --- | --- | --- | --- | --- | ------------ | ------- | --- | ---------------------- | --- | --- | -------- |
servicelife.Morestudiesareneededtoconfirmtheassociated
|     |     |     |     | failure   | mechanisms |          | and design  | the relevant | accelerated |            | tests. |
| --- | --- | --- | --- | --------- | ---------- | -------- | ----------- | ------------ | ----------- | ---------- | ------ |
|     |     |     |     | Moreover, |            | research | to separate | different    | failure     | mechanisms |        |
forcapacitorsandmagneticcomponentsinacceleratedtesting
|                      |                 |                   |                     | is still | at an   | early stage. |            |      |          |     |       |
| -------------------- | --------------- | ----------------- | ------------------- | -------- | ------- | ------------ | ---------- | ---- | -------- | --- | ----- |
| Fig. 5. Illustration | of the lifetime | model application | boundaries in terms |          |         |              |            |      |          |     |       |
|                      |                 |                   |                     | 2)       | Testing | With         | Conditions | More | Relevant | to  | S and |
of stress level and service life. SL and SH are the lower and higher stress H
| boundaries in           | which the lifetime | model is valid;       | L1 and L2 are the lower |         |              |             |            |           |              |             |          |
| ----------------------- | ------------------ | --------------------- | ----------------------- | ------- | ------------ | ----------- | ---------- | --------- | ------------ | ----------- | -------- |
|                         |                    |                       |                         | S L :   | As discussed | in          | [4], since | 1994,     | most studies |             | on power |
| and higher service      | life boundaries    | in which the specific | failure mechanism       |         |              |             |            |           |              |             |          |
|                         |                    |                       |                         | cycling | have         | focused     | on the     | stressors | of junction  | temperature |          |
| described bythelifetime | modelisdominant.   |                       |                         |         |              |             |            |           |              |             |          |
|                         |                    |                       |                         | swing   | T           | larger than | 40         | ◦ C. In   | comparison,  | the         | impact   |
j
4) Limited numbers of failure mechanisms are considered. of the heating time on the power cycling capability is less
◦
Theassociatedfailuremechanismsofthemodels(1)–(3) investigated. T is usually less than 40 C in the normal
j
represent a fraction. Failure during the early-to-middle operationofapowerconverter.Therearelinefrequencypower
service life could be responsible for a considerable cycles(i.e., 20 or 16.7ms) and lower frequencycyclesdue to
percentage, as revealed by the field experiences in [19] changes in load and ambient conditions.
and the discussions on power cycling test in [62]. Severalstudieshaveaimedtoaddresstheselimitations[57],
Other factors may also play an important role, such [66]–[71]. The power cycling is conducted with the heating
as humidity or corrosive atmosphere. As discussed in timet aslowas70ms,T =70 ◦ C,andthemeanjunction
|     |     |     |     |     | on  |     |     | j   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     | =   | ◦   |     |     |     |     |
Section II, these models are useful for designing power temperature T jm 122 C for two types of IGBT modules
electronic systems to avoid the wear-out failure mecha- in [68] and [69]. The impact of t on the cycling capability
on
nisms becominglife-limiting factors. However,they are of aluminum (Al) wire bonds and chip solder joints within
only part of the picture for the reliability prediction the range of 70 ms and 60 s is modeled separately. In [57],
and reliability design. Other reliability tools are also a typeoftransfermoldedintelligentpowermodules(IPMs) is
|     |     |     |     |     |     |     |     |     | T  | = 80.8 | ◦   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
widelyapplied,suchas failuremodeandeffectanalysis tested with t on from 147.5 ms to 2.5 s, j C, and
◦
(FMEA), robustness design, protection, fault-tolerance, T =122 C. In [67], the tests are with t down to 10 ms,
|     |     |     |     | jm  |     |     |     |     | on  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | T  |     | ◦   |     | =   | ◦   |     |     |
and burn-in testing. j down to 23 C, and T jm 100 C. Fig. 6 shows that it
|     |     |     |     | lastsformorethan500millioncyclesforthetestwitht |     |     |     |     |     |     | =10 |
| --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
on
D. Implications on Component-Level Accelerated Testing ms. This study also reveals that the impact of t becomes
on
Time and relevance are the two key considerations in insignificantwhenitislessthan40ms.Theresultsfromother
component-level accelerated testing. On the one hand, for a testing groups are shown in Fig. 6 with a different T and
jm
|     |     |     |     |     |     |     |     | I   |     | T = | ◦   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
given testing method and stressor S, the testing time can be current per bond wire foot bf normalized to jm 100 C
|     |     |     |     |     | =   | 5.87 |     | T  |     |     | ◦   |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
reduced by increasing the stress. However, this may risk trig- and I A [67]. When is between 23 C and
|     |     |     |     |     | bf  |     |     | j   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
gering new failure mechanisms that do not appear in normal 27 ◦ C, 10 ◦ C reduction of T would increase the cycling
jm
operation if the stress level is above S H shown in Fig. 5. On capability more than ten times. The high number of cycle-
the other hand, testing under accelerated conditions close to to-failure and distinct slope changes imply that it is near the
normaloperationis verytime-consumingandnota preferable elasticdeformationrange,wheretheresultantdegradationdue
toT
| choice. |     |     |     |     | isnegligible.Instead,theabsolutejunctiontemperature |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
j
1) Testing for Relevant Failure Mechanism and Stressor plays a significant role in such a range. Power cycling tests
|     |     |     |     |     |     |     |     | T  | T  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
S: Much effort is made to power cycling for triggering with a combination of high j and low j are proposed
specific failure mechanisms individually, such as bond wire in [70] and [71] to investigate the impact of low T within
j
wear-out, chip solder fatigue, and baseplate solder fatigue. a realistic testing period. T is down to 32.8 ◦ C with T
|     |     |     |     |     |     |     |     | j   |     |     | jm  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
◦
This is implemented by setting proper testing conditions in of 56 C in three different testing stages. This reveals that
termsofjunctiontemperature,temperatureswing,andcycling the low T has a minor impact on Stages I and II of the
j
period [4]. Besides component degradation curve analysis, degradation curve shown in Fig. 3. This effect is manifested
the Weibull plot [54, Ch.5] can be used to verify if the when the testing samples reach Stage III.
time-to-failure data from accelerated testing are for a single For capacitors, the time for degradation testing is usually
failure mechanism or a mix of two or more. For a given onthescaleofthousandsofhours[55],[60].Theacceleration
failure mechanism, the slope of the unreliability curve (i.e., factor is relatively lower compared with many tests designed
shapingfactor)showninFig.4tendstobeconsistent.In[63], for power semiconductor switches. In [60], the technical
the Weibull plots based on the testing results under different note finds that the lifetime model of aluminum electrolytic
temperature swings have a shaping factor of 37.5 and 14.8, capacitors depends on the ranges of both temperature and

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6483
Fig.6. Normalized lifetime of1200-A IGBTmoduleswithIHM-Bpackage (140mm×190mm)frommanufacturer A and B withton ≤40ms(current
per bond foot Ibf = 5.87 A and mean junction temperature at the beginning of the test Tjm = 100 ◦ C, and Hartmann refers to the model in [66]) [67].
Nf:numberofcycles tofailure; Tj:amplitude ofjunction temperature swing;and Tjm:meanjunction temperature.
voltage. However, testing to verify the dependence is still not if this control strategy is applied. For the second category,
found in the literature. the effect of degradation-dependent ESR and capacitance is
3) Testing With Relevant Control Strategy for Stressor S: not considered in the lifetime model.
Empirical lifetime models (1)–(4) are usually derived from The control strategy for accelerated stressors affects the
testsundertwoormorestresslevels.Thestresscontrolstrate- obtained lifetime models and converter-level modeling meth-
gies have an implication on how to apply the derived models. ods. Compared with [73], the cycle-to-failure based on con-
ForpowercyclingofIGBTmodules,thecontrolstrategywith stant T could be three times that of the standard power
j
a constantswitching-ON durationofdcpulsecurrentandcon- cycling control strategy.
stant switching-OFF duration is adopted by the standard IEC 4) Testing With Appropriate Sample Size: The sample size
60749-34(2011)[72]andECPEAQG324guide[53].T and that is used for testing is closely related to the confidence
jm
T usedforlifetimederivationarethevaluesatthebeginning interval. With the increase in sample size, the interval shown
j
of the testing. During the testing, they increase with possible in Fig. 4 between the 95% confidence level and 50% con-
parameter shifts due to degradation(e.g., thermal impedances fidence level becomes narrower. This implies a higher cer-
of thermal interface materials, ON-state voltage, and thermal tainty level. Typically, six to ten samples for each testing
resistance of baseplate solder and chip solder). In addition run are recommended for a meaningful statistical analysis
to this standard control strategy, another three methods (i.e., compromised with the required resources. For instance, ten
constant minimum and maximum case temperature, constant and 12 samples are used for capacitor and planar transformer
powerloss, andconstantT ) are also discussed in [73].The testing in [51] and [55], respectively. Nevertheless, it is
j
standard control strategy is most relevant to power electronic reported that 86% of the power cycling tests have only one
applications because parameter shifts can occur and result in sample per test run based on the 70 publications discussed
higherthermalstressesduringthe servicelifeofa powercon- in [4]. This implies that meaningful lifetime information is
verter. Therefore, the impact of these degradation-dependent unavailable in some cases even though a significant amount
parameters does not need to be considered for the lifetime of resources are used. Consequently, more attention to the
prediction in applications. If the lifetime model is obtained design of accelerated testing in terms of sample size is
from a power cycling test based on the other three control necessary.
strategies, then either part or all the parameter shifts need 5) Testing With Superimposed Stress Conditions: At the
to be included in the thermal stress modeling for lifetime application level, another aspect is the correlation between
prediction.Thiswouldincreasetheapplication-levelmodeling the testing under static stress level and field operation with
complexity. In this regard, the passive thermal cycling of dynamicenvironmentalandoperationalconditions.Twoques-
powermoduleswithoutcurrentstressignoresthisdegradation tionsneedtoberaisedhere:howcantheeffectoftwoormore
effect. different stress levels be combined? and does the dominant
Capacitortestingcanbeimplementedwithorwithoutripple failure mechanism remain the same under dynamic stress
current. For the first category, a dc bias voltage, ac ripple profiles?
current, and ambienttemperature are kept constant. Similarly, The degradation curve shown in Fig. 3 represents a single
thehot-spottemperatureandpeakvoltagestress bothincrease stress level. The accumulated damage model is used to ana-
during the test due to the changes in ESR and capacitance. lyze the joint damage of two or multiple stress levels. The
Thelifetime models(2)and (3)includethe degradationeffect time-to-failure corresponds to when the accumulated damage

6484 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
reaches one. The Palmgren–Miner rule that was presented IV. MISSION-PROFILE-BASED RELIABILITY PREDICTION
in 1945 [74] is a linear accumulated damage model that
In [1] and [2], the methods for reliability prediction are
is widely used for lifetime prediction. This rule assumes
reviewed. System-level reliability modeling methods, such as
no interaction between stress levels in terms of amplitude,
reliabilityblockdiagram(RBD),faulttreeanalysis(FTA),and
duration, and time sequence. Most of the wear-out failure
Markov analysis (MA), are compared in [2]. Much progress
predictionsofactivedevicesandpassivecomponentsinpower
hasbeenmadetowardreliabilitypredictionbasedonlong-term
electronics under dynamic stress conditions are based on
environmental and operational conditions (i.e., mission pro-
the Palmgren–Minerrule. The nonlinearaccumulateddamage
file). The methods and modeling procedures demonstrated
models that are used in other areas in reliability engineering
in [86] and [87] represent the state of the art for case studies
are summarized in [75]. The validity of the linear accumu-
on PV inverter and MMC, respectively.
lated damage model is questioned in [76]–[78]. Simulations
A mission-profile-based reliability prediction tool is devel-
in [76] show that the numbers of cycle-to-failure of bond
opedin [88].Thistoolisbasedon six modelingstepsstarting
wires are similar between the linear and nonlinear models.
fromsystem-levelmission profile to component-levelreliabil-
In[77],powercyclingtestsunderthreesetsof(T ,minimum
j ity modeling and system-level reliability modeling. This tool
junction temperature T , and t ) are performed first as the
jmin on featuresamodulardesignwithflexibleinputlevelsofmission
reference numbers of cycle-to-failure. The conditions at the
profiles.AsshowninFig.7,missionprofilesalongthemodel-
◦ ◦
beginning of the tests are Test 1: (100 C, 50 C, and 2 s);
ingprocessaredividedintofourlevels.Level1issystemlevel
◦ ◦ ◦ ◦
Test 2: (50 C, 100 C, and 1 s); and Test 3: (100 C, 50 C,
(e.g.,drivecyclesofacar,ambienttemperature,andhumidity).
and15s).Eachtesthassixsamples.Test4runsfor50%ofthe
Level 2 is the power electronic converter level, which can be
referencedlifetimeofTest1underthesameconditionandthen
directly measured or obtained from the system-level mission
for the rest of the time under the condition of Test 2 until the
profile modeling shown as Step 2 in Fig. 7 (e.g., derive
EOL reaches. Test 5 runs for 50% of the referenced lifetime
the output power, frequency, and voltage of a drive from
of Test 3 under the same condition and then until EOL under
the drive cycles, and motor models). Level 3 is down to
the condition of Test 2. Test 6 runs periodically with 20%
the component stresses of interest (e.g., the local ambient
of the reference lifetime of Test 1 and Test 2, respectively,
temperature, voltage, and current of a capacitor). Level 4 is
under the same conditions until failure. The results show
for stresses that are the variables in the lifetime models or
an average of 9.9%, 18.6%, and 0.6% higher numbers of
failure rate models used for component-level reliability mod-
cycle-to-failure from Test 4, Test 5, and Test 6, respectively,
eling. For dynamic thermal profiles, Rainflow counting [89]
to the values calculated based on the linear accumulated
is widely used to extract the periodic temperature swings
damagemodel.In[78],powercyclingtestswithsuperimposed
for the packaging-related wear-out of power semiconductor
stress levels are conductedwith more frequentchanges of the
switches. The use of Rainflow counting is discussed in detail
sequencesthanTest6in [77].Bondwire wear-outisassumed
in [86] and [87]. Depending on the available mission profile
asthedominantfailuremechanism[77],[78].Theconclusions
information, the tool runs part or all of the six steps. For
of both tests are consistent. A different conclusion may be
instance, if all the required information at Level 3 mission
drawn if the failure mechanism changesunder differentstress
profile is available, then the first three steps can be excluded,
levels,suchasthecasestudiesin[63].Itisworthwhilefurther
andthe softwaretoolstarts directlyfromStep 4. Fig. 8shows
investigating the validity of the linear accumulated damage
a capture of the graphic user interface (GUI) of the software
modelgiventhatthereportednumberofcasestudiesandtest-
tool customized for motor drive applications. The user can
ing conditions are still limited. Furthermore, no conclusions
select the available mission profilelevels, configurethe motor
are drawn to explain the observed results and the varieties in
and drive, and select the component, assembly, or system of
comparisons.In[79],anonlineardamagemodelforaluminum
interest for reliability analysis.
electrolytic capacitors is applied for the lifetime prediction in
The details of mission-profile-based reliability prediction
a motor drive application.
methods and the corresponding software tools can be found
Application-oriented testing under a representative mission
in [86]–[88]. A few additional aspects are discussed in the
profilecanbeusedtocheckifthedominantfailuremechanism
following.
remains the same with dynamic stress levels. In [80]–[82],
power modules and capacitors are tested with a sequence of
A. Mission Profile Data
stress levels emulating PV applications. The daily profile of
solar irradiance and ambient temperature is used [82] with a The mission profile represents all of the relevant stresses
scalingfactorto acceleratethetest. The purposeofthese tests that the item of interest withstands during the whole ser-
is not to derive a lifetime model because the results are the vice life, including stresses during manufacturing, burn-in-
joint effect of multiple stress levels in a single run. Instead, test, transportation, installation, and field operation. Lifetime
it is possible to verify the assumed failure mechanisms and and reliability prediction often uses limited information that
lifetime prediction results. To support such a superimposed is compromised by the available data and complexity. The
stress test, advanced mission profile emulators are necessary. limitations come from both the considered stressors and the
For example, in [83]–[85], the emulators for motor drives time span of the analyzed data. Predictions based on data
and multilevel modular converters (MMCs) are proposed, for the whole service life and models for all kinds of failure
respectively. mechanismsarerarelyfeasible.InthePVinvertercasestudies,

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6485
Fig.7. Six-stepmission-profile-based reliability prediction methodforpowerelectronic systems[88].
Fig.8. GUIofthedeveloped softwaretoolcustomizedformotordriveapplication withthesix-stepmodelingmethod[88].
annual solar irradiance and ambient temperature profiles are simplifymissionprofilesbyidentifyingrepresentativeprofiles.
used [86], [91]. In addition to the time span, the sampling Moreover, considering the varieties in thermal time constants
frequency is also critical, and this is relevant to the failure and failure mechanisms of different components (e.g., IGBT
mechanisms with different timescales. As studied in [92], modules and capacitors), the use of different data resolutions
the sampling frequency reveals significant variances in the can reduce the computational burden. A simplified mission
predicted lifetime based on wind speed data with different profile is usually sufficient because products are commonly
samplingfrequenciesforanoffshorewindapplication.Never- designed for applications with a wide range of operating
theless,guidelinesontheappropriatemissionprofileresolution conditions. However, for benchmarking or warranty analysis
are unavailable. of a specific application, a more detailed mission profile is
A mission profile logger is developed in [90] to collect preferred.
operationaland environmentaldata fromfield operations.The
B. Simplified Electrothermal Modeling
prototypeisshowninFig.9(a).Thismeasurestherawdataof
voltages and currents of a phase-leg and climatic conditions. Thermalmodelingisakeystepforthermal-relatedwear-out
The data canbe preprocessed,storedlocally,and collectedby failuremechanisms.Therearetwosignificantchallengeshere.
wirelessnetworks.Fig.9(b)showsanexampleofthemeasured 1) ThelocalambienttemperatureinLevel3missionprofile
climatic data inside and outside of the enclosure of a 60-kW shown in Fig. 7 is dependent on the cooling, layout,
PV inverter. power loss, and thermal couplings of the converter of
While the mission profile enables more realistic reliability interest. Therefore, system-level electrothermal model-
prediction, it increases the complexity and computational ing is needed to obtain the component-level thermal
burden, especially considering the dynamic thermal profile stress. In the case study of [86], the thermal stresses of
modeling[86],[87].Consequently,itcouldbemorefeasibleto capacitors and MOSFETs mainly depend on their local

6486 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
Fig.10. Two-daythermal stress profileofaPVmicroinverter withnatural
cooling [86]. Ta: ambient temperature outside the PV inverter enclosure;
Te: enclosure temperature; TjS1 , TjS3 , and TjS5 are the estimated junction
temperatureoftheMOSFETsoftheinverter;andTCdc:hot-spottemperature
ofthedc-linkcapacitor oftheinverter.
Fig. 9. Example of the mission profile logger and the obtained field
measurement data in [90]. (a) Photograph of the mission profile logger
prototype. (b) Temperature and humidity data logged from a 60-kW PV
inverterlocatedinthesolarparkinNordborg,Denmark.Thelegendindicates
the locations of the sensors Inside (In)/Outside (Out) Top (Tp) side and
Down(Dn)sideoftheinverter cabinet.
ambient temperatures (i.e., the enclosure temperature
andtheairtemperaturearoundthecomponents).Fig.10
◦
showsthattheenclosuretemperatureismorethan30 C
higher than the outside ambient temperature at heavy
loads. The maximum rises of component internal tem-
◦ ◦
peraturesduetoself-heatingare5 Cand15 C,respec-
tively, for the capacitor and MOSFETs. This implies Fig. 11. Example of system-level thermal modeling of an MMC [87].
(a) Prototype of the down-scaled MMC setup and its 24 submodules.
thattheaccuracyoftheenclosuretemperaturemodeling
(b) Measured ambient temperature profiles of the submodules with forced
is even more critical than that of the temperature rise air cooling. Tla1–Tla24 are the ambient temperature of the 24 submodules,
due to self-heating in this example. In another case respectively. Tla isthemaximumambienttemperature difference.
◦
study of MMC in [87], there is a maximum of 17 C
difference in local ambient temperatures among the modeling can identify hot-spots in the system. This
24submodules.Thesetupandmeasurementsareshown providesinsightsintoimprovingreliabilitybyheatredis-
in Fig. 11. This implies that there are likely to be tribution.
life-limiting submodules in the MMC setup. In [86] 2) Thermal modeling with long-term dynamic profiles is
and [87], converter-level finite element method (FEM) time-consuming. Therefore, in applications where the
simulations are performed to obtain the system-level ambient climatic conditions or loading conditions are
thermal models, including mutual coupling among the highly intermittent and dynamic (e.g., wind power, PV,
key heating sources. Moreover, system-level thermal and e-mobility), it is necessary to consider the impact

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6487
| of thermal  |           | capacitance. |             | A frequency-domain |                  |                 | thermal    |     |     |     |     |     |
| ----------- | --------- | ------------ | ----------- | ------------------ | ---------------- | --------------- | ---------- | --- | --- | --- | --- | --- |
| modeling    | method    |              | is proposed |                    | in [93].         | This            | simplifies |     |     |     |     |     |
| the         | modeling  | at           | different   | timescales         |                  | by coordinating |            |     |     |     |     |     |
| different   | thermal   | time         | constants   |                    | and the          | frequencies     | of         |     |     |     |     |     |
| power       | loss      | profiles.    | In          | [94],              | a simplification |                 | method     |     |     |     |     |     |
| is proposed |           | for thermal  |             | modeling           | with             | periodic        | power      |     |     |     |     |     |
| loss        | profiles. | An           | analytical  | error              | model            | is derived      | for        |     |     |     |     |     |
differentdegreesofsimplifications.Becausetheambient
| temperature |              | change      | is relatively |             | slow, | IGBT    | module    |     |     |     |     |     |
| ----------- | ------------ | ----------- | ------------- | ----------- | ----- | ------- | --------- | --- | --- | --- | --- | --- |
| junction    | temperatures |             | are           | assumed     | to    | change  | simulta-  |     |     |     |     |     |
| neously     | with         | the ambient |               | temperature |       | in [93] | and [94]. |     |     |     |     |     |
Nevertheless,thethermaltimeconstantsofcapacitorsor
largeheatsinksrangefromafewminutestohours,which
| are        | comparable | with       | ambient      |           | temperature       |                | dynamics. |     |     |     |     |     |
| ---------- | ---------- | ---------- | ------------ | --------- | ----------------- | -------------- | --------- | --- | --- | --- | --- | --- |
| A modified |            | thermal    | model        | is        | proposed          | to incorporate |           |     |     |     |     |     |
| the effect |            | of ambient | temperature  |           | dynamics          |                | [95]. The |     |     |     |     |     |
| snap-in    | aluminum   |            | electrolytic | capacitor |                   | that is        | used for  |     |     |     |     |     |
| the case   | study      | is         | shown        | in Fig.   | 12(a).            | The            | compara-  |     |     |     |     |     |
| tive       | results    | with the   | experimental |           | characterizations |                | are       |     |     |     |     |     |
| shown      | in Fig.    | 12(b).     | With         | the       | application       | of             | SiC and   |     |     |     |     |     |
GaNdevices,newchallengescomefromthenonlinearity
| of thermal        |         | impedance    | in                    | a wide                | range       | of temperature |         |     |     |     |     |     |
| ----------------- | ------- | ------------ | --------------------- | --------------------- | ----------- | -------------- | ------- | --- | --- | --- | --- | --- |
| stresses          | [96],   | [97],        | as well               | as PCB                | board-level |                | thermal |     |     |     |     |     |
| modeling          | and     | optimization |                       | [98].                 |             |                |         |     |     |     |     |     |
| C. Limitations    | of      | Existing     | Mission-Profile-Based |                       |             | Reliability    |         |     |     |     |     |     |
| Prediction        | Methods |              |                       |                       |             |                |         |     |     |     |     |     |
| The limitations   |         | of the       | existing              | mission-profile-based |             |                | relia-  |     |     |     |     |     |
| bility prediction |         | methods      | are discussed         |                       | in the      | following.     |         |     |     |     |     |     |
| 1) Limited        | failure | mechanisms   |                       |                       | and failure | sources        | are     |     |     |     |     |     |
considered. The discussed hardware failure is mainly Fig. 12. Proposed thermal modeling method considering the dynamics of
for the thermal-related wear-out. Other failures due to ambient temperature in [95]. (a) Capacitor testing sample and its thermal
|     |     |     |     |     |     |     |     | model parameters. | R1, R2, | C1, and C2 are the thermal | resistances | and |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | ------- | -------------------------- | ----------- | --- |
humidity, vibration, and contamination are not usually thermalcapacitances, respectively.(b)Experimentalverificationofthemodel
quantitatively analyzed. Sudden hardware failures and consideringtheimpactofambienttemperaturedynamics. Th isthecapacitor
software failures are yet to be included. For example, hot-spot temperature, Ta is the capacitor ambient temperature, and Ploss is
thepowerlossinthecapacitor.
| single-event-burnout |     |     | due | to cosmic | rays | is relevant | for |     |     |     |     |     |
| -------------------- | --- | --- | --- | --------- | ---- | ----------- | --- | --- | --- | --- | --- | --- |
electronic device applications at high elevations [99]. level. In practice, failure causes could be mutually
2) There is a lack of humidity aspect modeling. affected by multiple factors. The assumption of the iso-
Asdiscussedpreviously,humidity-relatedfailuremecha- lated failure of each component needs to be rigorously
nisms [21] may become dominant, especially for appli- justified [101]. In [102], a Markov reliability modeling
cations with intermittent loading profiles and frequent method is applied for multiphase dc–dc converters to
shutdownsandstart-ups.Atransienthygrothermalmod- analyze different failure scenarios.
| eling | method | is  | proposed | in  | analogy | with | electrical |               |                 |             |                |     |
| ----- | ------ | --- | -------- | --- | ------- | ---- | ---------- | ------------- | --------------- | ----------- | -------------- | --- |
|       |        |     |          |     |         |      |            | 5) Activities | in system-level | reliability | demonstrations | are |
domain models in [100]. It is, therefore, worthwhile expanding but still limited in the literature. In [103]
investigating the application of such models in power and[104],converter-levelreliabilitytestingisperformed
electronic converters. for motor drives and PV inverters, respectively. The
3) The linear accumulated damage model is dominantly system-leveltestingoutcomescouldidentifytheweakest
| used | in wear-out |     | failure | prediction. | To  | what | extent | it    |                 |                      |            |      |
| ---- | ----------- | --- | ------- | ----------- | --- | ---- | ------ | ----- | --------------- | -------------------- | ---------- | ---- |
|      |             |     |         |             |     |      |        | links | and demonstrate | that the reliability | is no less | than |
is a reasonable approximation is still open to question. the designed target.
| Various | nonlinear |     | damage | models | are | available, | as dis- |     |                            |     |     |     |
| ------- | --------- | --- | ------ | ------ | --- | ---------- | ------- | --- | -------------------------- | --- | --- | --- |
|         |           |     |        |        |     |            |         | V.  | OUTLOOKFORPOWERELECTRONICS |     |     |     |
cussedin[75].However,therelevanceandthenecessity
|        |      |         |            |     |     |     |     |     | RELIABILITY | RESEARCH |     |     |
| ------ | ---- | ------- | ---------- | --- | --- | --- | --- | --- | ----------- | -------- | --- | --- |
| to use | them | are not | yet known. |     |     |     |     |     |             |          |     |     |
4) Failure interactions among components and external Power electronics reliability research has been booming in
factors are not considered in the system-level reliability recent years, both at universities and in companies. Among
analysis.Asshownin[86]and[87],theRBDmethodis others, design-margin reduction, predictive maintenance, cost
appliedto calculatethe system-levelreliabilitybasedon of failure, and the increasing use of power electronic con-
the component-level reliability and system redundancy vertersin critical applicationsare the main drivers. Life-cycle

6488 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
performance optimization and cost reduction are of interest, special operation state (e.g., startup transient [109]),
as are the time-zero performance and design cost. The recog- and system-level health precursors (e.g., efficiency) are
nition of the importance of data also plays a significant role among the most promisingmethods. In [110],a digital-
in reliability studies. This section will describe several future twin-based condition monitoring concept is proposed,
research opportunities. representing a future direction of combining physi-
|                           |     |           |     |           |            |             |     | cal         | power | electronic  | circuit |           | models     | and data-driven   |
| ------------------------- | --- | --------- | --- | --------- | ---------- | ----------- | --- | ----------- | ----- | ----------- | ------- | --------- | ---------- | ----------------- |
|                           |     |           |     |           |            |             |     | approaches. |       | A composite |         | health    | precursor  | is proposed       |
| A. Physics-of-Degradation |     |           | and | Condition | Monitoring |             |     |             |       |             |         |           |            |                   |
|                           |     |           |     |           |            |             |     | in [111]    | by    | an optimal  |         | weighting | of         | multiple relevant |
| The failure-free          |     | operation | may | become    | a          | requirement | for |             |       |             |         |           |            |                   |
|                           |     |           |     |           |            |             |     | parameters. |       | This method |         | expects   | to improve | the sensi-        |
power electronic converters as they are increasingly used in tivity and reduce noise in the remaining useful lifetime
| reliability-, | availability-, |     | or safety-criticalapplications. |     |     |     | If fail- |     |     |     |     |     |     |     |
| ------------- | -------------- | --- | ------------------------------- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
predictionbasedonmultiplesourcesofdegradationdata.
| ure is not | an  | option, | then the | understanding |     | and modeling |     |            |              |     |            |     |           |            |
| ---------- | --- | ------- | -------- | ------------- | --- | ------------ | --- | ---------- | ------------ | --- | ---------- | --- | --------- | ---------- |
|            |     |         |          |               |     |              |     | Artificial | intelligence |     | (AI)-based |     | condition | monitoring |
of the degradation process become important. A degradation methods [112] hold great potential with the continued
| curve retains | all | the information |     | from | accelerated | degrada- |     |              |     |                |     |         |     |                   |
| ------------- | --- | --------------- | --- | ---- | ----------- | -------- | --- | ------------ | --- | -------------- | --- | ------- | --- | ----------------- |
|               |     |                 |     |      |             |          |     | digitization |     | and increasing |     | amounts | of  | available data in |
tion testing, as shown in Fig. 3, while a lifetime model is power electronics applications.
| based on | a single | data    | point   | (i.e., EOL)   | of  | the degradation |     |     |     |     |     |     |     |     |
| -------- | -------- | ------- | ------- | ------------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| curve of | each     | testing | sample. | Understanding |     | the physics-of- |     |     |     |     |     |     |     |     |
degradationenablesidentifyingkeyfeaturesofthedegradation B. Accumulated Damage Modeling
process.
|     |     |     |     |     |     |     |     | Fig. 13 | illustrates | the | validity | of  | the linear | accumulated |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ----------- | --- | -------- | --- | ---------- | ----------- |
Condition monitoring is likely to become an even more damage model under different types of degradation curves.
importanttoolinreliabilityengineeringiffailureisnotallowed For the sake of simplicity, Fig. 13(a) and (c) shows the
| within the | service | life. | Degradation | modeling |     | is essential | to  |                      |     |     |        |        |       |               |
| ---------- | ------- | ----- | ----------- | -------- | --- | ------------ | --- | -------------------- | --- | --- | ------ | ------ | ----- | ------------- |
|            |         |       |             |          |     |              |     | degradationcurvesfor |     | two | stress | levels | A and | B, havingonly |
predictive maintenance. The following two key challenges Stage II and Stage III illustrated in Fig. 3, respectively. More
| remain | to be addressed. |     |     |     |     |     |     |               |       |     |               |     |        |              |
| ------ | ---------------- | --- | --- | --- | --- | --- | --- | ------------- | ----- | --- | ------------- | --- | ------ | ------------ |
|        |                  |     |     |     |     |     |     | general cases | could | be  | a combination |     | of the | curves shown |
1) Mostconditionmonitoringmethodsarelimitedtoasin- inFig.13(a)and(c).Eventhoughthedegradationcurveatthe
gle type of component or an individual component [6], beginningoftestingmaydiverse,asshownin[78],itisnegli-
[7], [13], [105]. Nevertheless, at the application level, gible because it is usually a short period of the whole service
degradation could occur on multiple components con- life.StressLevelsA andBcorrespondtothe referencedtime-
currently. Different sources of parameter shifts happen to-failure L A and L B , respectively. Three different scenarios
simultaneously. However, the mutual effects cannot be of the nonlinear degradation curve under Stress Level B are
verified based on the assumption that a single com- considered in Fig. 13(c), namely: Scenario 1, Scenario 2, and
ponent degrades at a time. Moreover, a full converter Scenario 3. Fig. 13(b) and (d) shows the degradation curves
or an assembly is usually replaced in the presence of with normalizedhour/cycle/mileageto L and L . As shown
|     |     |     |     |     |     |     |     |     |     |     |     |     | A   | B   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
component failure. It is, therefore, essential to monitor in Fig. 13(b), the degradation curves overlap under the two
the health status of the entire converter or the assem- stress levels due to the normalization of the x-axis. Scenario
bly parts of interest. Converter-level health precursors 1inFig.13(c)correspondstowhenthetwodegradationcurves
and converter-level signal measurements are alternative ofLevelAandLevelBarethesame.Scenario2andScenario
solutions. 3 represent the cases for two different degradation curves
2) Robustandcost-effectiveconditionmonitoringmethods showninFig.13(d).AllthecurvesreachEOLcriteriaat100%
arerequired.Eventhoughawiderangeofhealthprecur- with testing under the individual level of stress conditions.
sors and implementations is proposed in the literature, At x% of the time-to-failure, the stress level changes from A
| they | are | rarely adopted |     | in field | operation. | Complexity, |     | to B. |     |     |     |     |     |     |
| ---- | --- | -------------- | --- | -------- | ---------- | ----------- | --- | ----- | --- | --- | --- | --- | --- | --- |
cost, design constraint, accessibility, and effectiveness With the linear degradationcurves,the superimposedstress
under field operation environment are of great concern. of Level A and Level B results in the same time-to-failure as
In [106] and [107], challenges in circuit parasitics and predicted by the linear accumulated damage model, as shown
data analytics of a condition monitoring unit for IGBT inFig.13(b).Withthenonlineardegradationcurves,thehealth
power modules are addressed. A converter-level power precursor’s change at different intervals in the x-axis is
device ON-state voltage measurement method is devel- different due to the nonlinearity. For instance, the testing
oped in [108], which reduces the design cost and pro- under LevelA stress conditionat the very beginningcauses a
y
vides a converter-level plug-and-play solution without negligible change of compared with a significant change
the need to connect to the monitored power devices. in approaching the EOL L , as shown in Fig. 13(c). Never-
A
Condition monitoring provides an ancillary service, theless, this nonlinearity in the degradation curve is already
which is critical to minimize the new risk brought by considered in lifetime models, which does not necessarily
theaddedsensors,measurementcircuits,and/orsoftware imply that a nonlinear accumulated damage model is needed.
algorithms.Inaddition,thisservicemayonlyrequirean In Scenario 1 for Stress Level B, its degradation curve is the
intermittentoperation.ItisneededmostduringStageIII same as that of Stress A in Fig. 13(d). The resultant changes
y
of the degradation curve shown in Fig. 3. Anomaly of under Stress Levels A and B are the same at x% of
detection from similar systems in operation, leverage operation. The degradation curve ends at 100% under Stress

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6489
Fig. 13. Illustration for accumulated damage based on degradation curves under a sequence of two stress levels for a given failure mechanism. The
normalization of the x-axis in (b)and (d) is with respect to the time-to-failure LA and LB, respectively, for the curves under stress Level A and Level B.
(a)Lineardegradation curves. (b)Normalized linear degradation curves.(c)Nonlineardegradation curves.(d)Normalized nonlinear degradation curves.
Level B of operation after x%. In this scenario, the linear C. Humidity-Induced Failure Mechanisms, and Failure
accumulateddamagemodelisvalid.InScenario2,thechange Mechanisms of SiC MOSFETs, GaN devices, and Magnetic
ofy
| atx%ofoperationunderStressLevelAisequaltothat |               |     |         |     |                      |     |     | Components                                           |     |     |     |     |     |     |
| --------------------------------------------- | ------------- | --- | ------- | --- | -------------------- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| until x                                       | % underStress |     | LevelB. | The | degradationcurvefrom |     |     |                                                      |     |     |     |     |     |     |
| B2                                            |               |     |         |     |                      |     |     | Thelevelofattentiontothehumidity-inducedfailuremech- |     |     |     |     |     |     |
x%ofoperationunderStressLevelBisequivalenttoshiftthe
anismsofpowerelectronicsystemsisrelativelylowerthanthat
degradation curve from x B2 %, resulting in a time-to-failure for electrothermal and thermalmechanical induced failures.
| more than | 100%, | which | is higher | than | that | predicted | by the |               |      |             |          |     |          |               |
| --------- | ----- | ----- | --------- | ---- | ---- | --------- | ------ | ------------- | ---- | ----------- | -------- | --- | -------- | ------------- |
|           |       |       |           |      |      |           |        | Nevertheless, | they | are closely | relevant |     | to field | applications, |
linear accumulated damage model. Similarly, in Scenario 3, as discussed in Section II. Consequently, more research on
itcanbederivedthatthesuperimposedstressresultsinatime-
|            |      |            |     |     |     |     |     | modeling | of humidity          | impact | and     | microclimatic |     | design and |
| ---------- | ---- | ---------- | --- | --- | --- | --- | --- | -------- | -------------------- | ------ | ------- | ------------- | --- | ---------- |
| to-failure | less | than 100%. |     |     |     |     |     |          |                      |        |         |               |     |            |
|            |      |            |     |     |     |     |     | control  | for power electronic |        | systems | is suggested. |     |            |
According to Fig. 13, it is possible to check if the linear Even with increasing studies on the failure mechanisms
accumulateddamagemodelisvalidbyanalyzingthedegrada-
|     |     |     |     |     |     |     |     | and accelerated | testing | of  | SiC and | GaN | switches | [10], [11], |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | ------- | --- | ------- | --- | -------- | ----------- |
tioncurvesfromthetestingunderindividualstressconditions. the relevance between these component-level investigations
| For instance, |         | the degradation |                    | curves | presented | in  | [77] feature |                                   |     |     |     |               |              |     |
| ------------- | ------- | --------------- | ------------------ | ------ | --------- | --- | ------------ | --------------------------------- | --- | --- | --- | ------------- | ------------ | --- |
|               |         |                 |                    |        |           |     |              | and converter-levelapplicationsis |     |     |     | still largely | unknown.SiC- |     |
| a large       | portion | of              | linear degradation |        | shown     | in  | Fig. 13(a)   |                                   |     |     |     |               |              |     |
andGaN-basedindustrialsystemsarelimitedattheearlystage
and a sharp and short period of nonlinear degradation shown of its operation. It is, therefore, expected that the commercial
| in Fig. 13(c).Unlike |     |        | the case    | shown | in Fig.   | 13(a),the | bound-      |            |              |               |           |               |     |              |
| -------------------- | --- | ------ | ----------- | ----- | --------- | --------- | ----------- | ---------- | ------------ | ------------- | --------- | ------------- | --- | ------------ |
|                      |     |        |             |       |           |           |             | use of     | SiC devices  | in automotive |           | applications  |     | will provide |
| ary between          | the | linear | degradation | and   | nonlinear |           | degradation |            |              |               |           |               |     |              |
|                      |     |        |             |       |           |           |             | more field | experiences. | The           | decreased | short-circuit |     | withstand    |
y.
| could correspond |     | to  | different |     | This | implies | a slightly |         |             |         |         |     |            |         |
| ---------------- | --- | --- | --------- | --- | ---- | ------- | ---------- | ------- | ----------- | ------- | ------- | --- | ---------- | ------- |
|                  |     |     |           |     |      |         |            | time of | SiC and GaN | devices | demands |     | protection | schemes |
different slope of the degradation curves for Stress Levels A sub-μs 1–2-μs
|     |     |     |     |     |     |     |     | with | or  | decision-making |     |     | and action. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --------------- | --- | --- | ----------- | --- |
andBinFig.13(b).AccordingtoFig.3,thedegradationcurve Theunderstandingoffailuremechanismsofmagneticcom-
| consisting     | of  | Stage      | III can be   | assumed | linear | within | a short     |               |                     |     |              |          |          |              |
| -------------- | --- | ---------- | ------------ | ------- | ------ | ------ | ----------- | ------------- | ------------------- | --- | ------------ | -------- | -------- | ------------ |
|                |     |            |              |         |        |        |             | ponents       | in power electronic |     | applications |          | is still | limited. The |
| time interval. |     | Therefore, | as indicated |         | by the | tests  | in [78] and |               |                     |     |              |          |          |              |
|                |     |            |              |         |        |        |             | early studies | presented           | in  | [51]         | and [52] | show     | the need     |
the sixth test in [77], the linear accumulated damage model to further investigate the failure mechanisms and degrada-
| is valid | for the | cases | under more | frequent |     | changes | of stress |              |            |     |        |          |            |     |
| -------- | ------- | ----- | ---------- | -------- | --- | ------- | --------- | ------------ | ---------- | --- | ------ | -------- | ---------- | --- |
|          |         |       |            |          |     |         |           | tion models, | especially | for | planar | magnetic | components | and |
levels, which differs from the fourth and fifth tests in [77]. medium-voltage medium-frequencyisolation transformers.
| The perspectives |     | provided | in  | Fig. 13 | are based | on  | theoretical |     |     |     |     |     |     |     |
| ---------------- | --- | -------- | --- | ------- | --------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
analysis and are consistent with superimposed stress testing D. High-Resolution and Fast-Temperature Measurement
observations, focusing on the Palmgren–Miner rule’s validity. Thermal stress is related to various failure mecha-
However, more tests to verify the assumptions are necessary nisms. Temperature sensitive electrical parameters (TSEPs)
in the future. are widely studied for temperature estimation, limited to

6490 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
|     |     |     |     |     |     |     | models | used | in  | the digital | platform. |     | Since | a significant |
| --- | --- | --- | --- | --- | --- | --- | ------ | ---- | --- | ----------- | --------- | --- | ----- | ------------- |
partistobedonebysimulations,itispossibletoreduce
therequiredtimewithoutcompromisingtherelevanceof
the results.
|     |     |     |     |     |     |     | 2) Accelerated              |           | testing        | combined                   |              | with        | early       | wear-out pre-  |
| --- | --- | --- | --- | --- | --- | --- | --------------------------- | --------- | -------------- | -------------------------- | ------------ | ----------- | ----------- | -------------- |
|     |     |     |     |     |     |     | diction.                    |           | The prediction |                            | for          | degradation |             | curves based   |
|     |     |     |     |     |     |     | on                          | early     | testing        | results                    | is           | applied     | for         | film capaci-   |
|     |     |     |     |     |     |     | tors                        | [115]     | and batteries  |                            | [116],       | [117].      | In the      | recent case    |
|     |     |     |     |     |     |     | study                       | [117],    | it has         | been                       | demonstrated |             | that        | the required   |
|     |     |     |     |     |     |     | time                        | for       | identifying    | high-cycle-life            |              |             | charging    | protocols      |
|     |     |     |     |     |     |     | among224batterycandidatesis |           |                |                            |              | reducedto   |             | 3.2%ofthe      |
|     |     |     |     |     |     |     | expected                    |           | time without   |                            | early        | prediction. | It          | is, therefore, |
|     |     |     |     |     |     |     | worth                       | exploring |                | the concept                |              | of the      | degradation | testing        |
|     |     |     |     |     |     |     | of                          | power     | electronic     | components                 |              | and         | converters. |                |
|     |     |     |     |     |     |     | F. Reliability              | Modeling  |                | of Power-Electronics-Based |              |             |             | Power          |
Systems
|     |     |     |     |     |     |     | Theimpactofpowerelectronicconvertersonthe |     |     |     |     |     |     | reliability |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- |
ofpowersystemsisstillanopenquestionwiththeirincreasing
penetrationlevel.Hundredsandthousandsofpowerelectronic
|     |     |     |     |     |     |     | converters          | may           | be used                   | in a             | local | electrical  | network.        | Conse-         |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | ------------- | ------------------------- | ---------------- | ----- | ----------- | --------------- | -------------- |
|     |     |     |     |     |     |     | quently,reliability |               | performanceplaysa         |                  |       | role        | in the          | availability   |
|     |     |     |     |     |     |     | and security        | of            | the electric              | network.         |       | The         | interconnects,  | inter-         |
|     |     |     |     |     |     |     | actions,            | and control   | dynamics                  |                  | among | different   |                 | converters all |
|     |     |     |     |     |     |     | add extra           | layers        | of complexity.            |                  | This  | brings      | new             | challenges     |
|     |     |     |     |     |     |     | in scaling          | up            | the mission-profile-based |                  |       | reliability |                 | prediction     |
|     |     |     |     |     |     |     | methods             | discussed     | in                        | Section          | IV.   | FMEA        | is a            | useful tool to |
|     |     |     |     |     |     |     | prioritize          | the available |                           | resources        | to    | deal        | with the        | top-ranked     |
|     |     |     |     |     |     |     | failure causesin    |               | a system.                 | Consequently,new |       |             | multitime-scale |                |
modelingapproachesareneededtosimplifythemodelingpro-
|     |     |     |     |     |     |     | cedures,        | meanwhile | maintaining    |     | reasonable         |     | prediction | accu-      |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --------- | -------------- | --- | ------------------ | --- | ---------- | ---------- |
|     |     |     |     |     |     |     | racy. Real-time |           | simulation     | of  | large-scale        |     | systems    | could also |
|     |     |     |     |     |     |     | becomefeasible  |           | forreliability |     | study,extendingits |     |            | scope from |
Fig. 14. Thermoreflectance measurement method and an example of the electromagnetic domain.
| the measured           | temperature | profile      | proposed       | in [113].      | (a) Schematic | of the         |     |     |     |                |     |     |     |     |
| ---------------------- | ----------- | ------------ | -------------- | -------------- | ------------- | -------------- | --- | --- | --- | -------------- | --- | --- | --- | --- |
| thermoreflectance      |             | measurement  | configuration, | including      | the           | charge-coupled |     |     |     |                |     |     |     |     |
| device (CCD)           | detector,   | illumination | source,        | microscope     | objective,    | and the        |     |     |     |                |     |     |     |     |
|                        |             |              |                |                |               |                |     |     |     | VI. CONCLUSION |     |     |     |     |
| temperature-controlled |             | stage.       | (b)Measured    | drain current, | voltage       | and calcu-     |     |     |     |                |     |     |     |     |
lated power dissipation, temperature based on measurement and simulation This article reviews the latest power electronics reliability
of the CGH40025F Cree GaN HEMT operating in a class-A amplifier, and research in field experiences, failure mechanisms,
| temperature | measuredwithatemporal |     |     | resolution of50ns. |     |     |                 |     |     |           |     |         |          |     |
| ----------- | --------------------- | --- | --- | ------------------ | --- | --- | --------------- | --- | --- | --------- | --- | ------- | -------- | --- |
|             |                       |     |     |                    |     |     | component-level |     | EOL | criteria, |     | testing | methods, | and |
100-μs temporal resolution [114]. In [113], a noninvasive system-level mission-profile-based reliability prediction
|     |     |     |     |     |     |     | methods. | The | discussions |     | on failure |     | mechanisms | have |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | ----------- | --- | ---------- | --- | ---------- | ---- |
thermalmeasurementtechniquecalledthermoreflectancether-
mography is proposed with nanosecond time and submi- focused on the short circuits of SiC MOSFETs and GaN
crometer spatial resolutions. This technique is promising for devices and the magnetic components. Perspectives on
|     |     |     |     |     |     |     | the degradation |     | curve, | lifetime | definition, |     | lifetime | model |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------ | -------- | ----------- | --- | -------- | ----- |
temperaturemeasurementduringshortcircuitsandMHz-level
high-frequency converters. Fig. 14(a) shows a schematic of applicability boundary, and linear accumulated damage
the measurement setup. Fig. 14(b) shows the measured tem- model validity are provided. In addition, the limitations in
peratureprofileswithatemporalresolutionof50ns.However, existing lifetime models and mission profile-based reliability
morestudiesarenecessarytoliftthematuritylevelandextend prediction methods are discussed. Finally, future research
|              |               |             |         |          |         |             | demands            | and       | opportunities |                  | are         | identified: |         | 1) physics-  |
| ------------ | ------------- | ----------- | ------- | -------- | ------- | ----------- | ------------------ | --------- | ------------- | ---------------- | ----------- | ----------- | ------- | ------------ |
| it to other  | applications. |             |         |          |         |             |                    |           |               |                  |             |             |         |              |
|              |               |             |         |          |         |             | of-degradation     |           | and condition |                  | monitoring; |             | 2)      | accumulated  |
| E. Reduction | of            | Reliability | Testing | Time     |         |             |                    |           |               |                  |             |             |         |              |
|              |               |             |         |          |         |             | damage             | modeling; | 3)            | humidity-induced |             |             | failure | mechanisms,  |
| It is        | necessary     | to reduce   | the     | required | time of | accelerated |                    |           |               |                  |             |             |         |              |
|              |               |             |         |          |         |             | failure mechanisms |           |               | of SiC           | MOSFETs,    |             | GaN     | devices, and |
degradation testing and reliability demonstration testing. Two magneticcomponents;4)high-resolutionandfast-temperature
promising directions are briefly discussed in the following. measurement; 5) reduction of reliability testing time; and
1) Virtual-Physical Hybrid Testing Method: A digital plat- 6) reliability modeling of power-electronics-based power
form can be built to simulate the degradation of power systems. For further backgroundinformationon these aspects
electronic components under accelerated or normal and other relevant research topics, we refer the reader
| stress | levels. | Physical | testing | is used | to  | calibrate the | to [1]–[13]. |     |     |     |     |     |     |     |
| ------ | ------- | -------- | ------- | ------- | --- | ------------- | ------------ | --- | --- | --- | --- | --- | --- | --- |

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6491
REFERENCES [25] C. Herold, J. Sun, P. Seidel, L. Tinschert, and J. Lutz, “Power
cycling methods forSiC MOSFETs,”in Proc. 29th Int. Symp. Power
[1] H. Wang, M. Liserre, and F. Blaabjerg, “Toward reliable power Semiconductor Devices IC’s(ISPSD),May2017,pp.367–370.
electronics: Challenges, design tools, and opportunities,” IEEE Ind. [26] B. Huet al., “Failure and reliability analysis ofa SiCpower module
Electron. Mag.,vol.7,no.2,pp.17–26,Jun.2013. basedonstresscomparisontoaSidevice,”IEEETrans.DeviceMater.
[2] H. Wang et al., “Transitioning to physics-of-failure as a reliability Rel.,vol.17,no.4,pp.727–737,Dec.2017.
driver in power electronics,” IEEEJ. Emerg.Sel. Topics Power Elec- [27] T.-T. Nguyen, A. Ahmed, T. V. Thang, and J.-H. Park, “Gate oxide
tron.,vol.2,no.1,pp.97–114,Mar.2014. reliabilityissuesofSiCMOSFETsundershort-circuitoperation,”IEEE
[3] H.S.-H.Chung,H.Wang,F.Blaabjerg,andM.Pecht,eds,Reliability Trans.PowerElectron.,vol.30,no.5,pp.2445–2455,May2015.
of Power Electronic Converter Systems. London, U.K.: Institution of [28] G.Romanoetal.,“Acomprehensivestudyofshort-circuitruggedness
Engineering andTechnology, 2015. ofsiliconcarbidepowerMOSFETs,”IEEEJ.Emerg.Sel.TopicsPower
[4] C.Durand,M.Klingler,D.Coutellier, andH.Naceur,“Powercycling Electron.,vol.4,no.3,pp.978–987,Sep.2016.
reliability of power module: A survey,” IEEE Trans. Device Mater. [29] P.D.Reigosa, F.Iannuzzo, H.Luo,andF.Blaabjerg, “Ashort-circuit
Rel.,vol.16,no.1,pp.80–97,Mar.2016. safe operation area identification criterion for SiC MOSFET power
[5] N.Baker,M.Liserre,L.Dupont,andY.Avenas,“Improvedreliability modules,” IEEE Trans. Ind. Appl., vol. 53, no. 3, pp.2880–2887,
ofpower modules:A review ofonline junction temperature measure- May2017.
ment methods,” IEEE Ind. Electron. Mag., vol. 8, no. 3, pp.17–27, [30] E.-P.Eniet al., “Short-circuit degradation of 10-kV 10-A SiC MOS-
Sep.2014. FET,” IEEE Trans. Power Electron., vol. 32, no. 12, pp.9342–9354,
[6] Y. Avenas, L.Dupont, N.Baker, H.Zara, and F.Barruel, “Condition Dec.2017.
monitoring: A decade of proposed techniques,” IEEE Ind. Electron. [31] Z.Wangetal.,“Temperature-dependent short-circuitcapability ofsil-
Mag.,vol.9,no.4,pp.22–36,Dec.2015. iconcarbidepowerMOSFETs,”IEEETrans.PowerElectron.,vol.31,
[7] H. Oh,B.Han, P.McCluskey, C. Han, andB. D.Youn,“Physics-of- no.2,pp.1555–1566,Feb.2016.
failure,conditionmonitoring,andprognosticsofinsulatedgatebipolar
[32] H. Du, P. Diaz Reigosa, L. Ceccarelli, and F. Iannuzzo, “Impact of
transistor modules: A review,” IEEE Trans. Power Electron., vol. 30,
repetitiveshort-circuittestsonthenormaloperationofSiCMOSFETs
no.5,pp.2413–2426,May2015.
considering case temperature influence,” IEEE J. Emerg. Sel. Topics
[8] R.Khazaka,L.Mendizabal,D.Henry,andR.Hanna,“Surveyofhigh- PowerElectron., vol.8,no.1,pp.195–205,Mar.2020.
temperature reliability of power electronics packaging components,”
[33] P.D.Reigosa,H.Luo,andF.Iannuzzo,“Implicationsofageingthrough
IEEETrans.PowerElectron.,vol.30,no.5,pp.2456–2464,May2015.
power cycling on the short-circuit robustness of 1.2-kV SiC MOS-
[9] H. Lee, V. Smet, and R. Tummala, “A review of SiC power module
FETs,”IEEETrans.PowerElectron.,vol.34,no.11,pp.11182–11190,
packaging technologies: Challenges, advances, and emerging issues,”
Nov.2019.
IEEEJ.Emerg.Sel.TopicsPowerElectron.,vol.8,no.1,pp.239–255,
[34] S.Jietal.,“Short-circuitcharacterization andprotectionof10-kVSiC
Mar.2020.
mosfet,”IEEETrans.PowerElectron.,vol.34,no.2,pp.1755–1764,
[10] Z. Ni, X. Lyu, O. P. Yadav, B. N. Singh, S. Zheng, and D. Cao,
Feb.2019.
“Overview of real-time lifetime prediction and extension for SiC
[35] D.-P. Sadik et al., “Short-circuit protection circuits for silicon-
power converters,” IEEE Trans. Power Electron., vol. 35, no. 8,
carbide powertransistors,” IEEETrans.Ind.Electron., vol. 63,no.4,
pp.7765–7794,Aug.2020.
pp.1995–2004,Apr.2016.
[11] M. Meneghini et al., “Reliability and failure analysis in power GaN-
[36] D.Rothmund,D.Bortis,andJ.W.Kolar,“Highlycompactisolatedgate
HEMTs: An overview,” in Proc. IEEE Int. Rel. Phys. Symp. (IRPS),
driverwithultrafastovercurrentprotectionfor10kVSiCMOSFETs,”
Apr.2017,pp.3B–2.1–3B–2.8.
CPSS Trans. Power Electron. Appl., vol. 3, no. 4, pp.278–291,
[12] H.WangandF.Blaabjerg,“ReliabilityofcapacitorsforDC-linkappli-
Dec.2018.
cationsinpowerelectronicconverters—Anoverview,”IEEETrans.Ind.
Appl.,vol.50,no.5,pp.3569–3578,Oct.2014. [37] K. Sun, J. Wang, R. Burgos, and D. Boroyevich, “Design, analysis,
anddiscussionofshortcircuitandoverloadgate-driverdual-protection
[13] Z.Zhao,P.Davari,W.G.Lu,H.Wang,andF.Blaabjerg,“Anoverview
schemefor1.2-kV,400-ASiCMOSFETmodules,”IEEETrans.Power
ofcondition monitoring techniques forcapacitors inDC-linkapplica-
Electron.,vol.35,no.3,pp.3054–3068,Mar.2020.
tions,”IEEETrans.PowerElectron.,earlyaccess,Sep.11,2020,doi:
10.1109/TPEL.2020.3023469. [38] P. Hofstetter and M.-M. Bakran, “The two-dimensional short-circuit
[14] M. R. Lyu et al., Handbook of Software Reliability Engineering. detection protection for SiC MOSFETs in urban rail transit applica-
Hightstown, NJ,USA:McGraw-Hill, 1996. tion,” IEEE Trans. Power Electron., vol. 35, no. 6, pp.5692–5701,
Jun.2020.
[15] L. Podofillini, “Human reliability analysis,” in Handbook of Safety
Principles. Hoboken, NJ,USA:Wiley,2018. [39] B.ZhangandS.Wang,“AsurveyofEMIresearchinpowerelectronics
[16] M.Ciappa,“Selected failure mechanismsofmodernpowermodules,” systems with wide-bandgap semiconductor devices,” IEEE J. Emerg.
Microelectron. Rel.,vol.42,nos.4–5,pp.653–667,Apr.2002. Sel.TopicsPowerElectron.,vol.8,no.1,pp.626–643,Mar.2020.
[17] P. Hacke et al., “A status review of photovoltaic power conversion [40] M.TˇapajnaandC.Koller,ReliabilityIssuesinGaNElectronicDevices,
equipment reliability, safety, andquality assurance protocols,” Renew. Chaper 6 of Nitride Semiconductor Technology, F. Roccaforte and
Sustain. EnergyRev.,vol.82,pp.1097–1112,Feb.2018. M.Leszczynski, Eds.Hoboken,NJ,USA:Wiley,2020.
[18] A. Golnas, “PV system reliability: An Operator’s perspective,” IEEE [41] M. A. González-Sentís, P. Tounsi, A. Bensoussan, and A. Dufour,
J.Photovolt., vol.3,no.1,pp.416–421,Jan.2013. “Degradation indicators of power-GaN-HEMT under switching
[19] K. Fischer et al., “Reliability of power converters in wind turbines: power-cycling,” Microelectron. Rel., vols. 100–101, Sep. 2019,
Exploratoryanalysisoffailureandoperatingdatafromaworldwidetur- Art.no.113412.
binefleet,”IEEETrans.PowerElectron.,vol.34,no.7,pp.6332–6344, [42] S. Song, S. Munk-Nielsen, C. Uhrenfeldt, and I. Trintis, “Failure
Jul.2019. mechanismanalysisofadiscrete650VenhancementmodeGaN-on-Si
[20] K.Fischeretal.,“Field-experiencebasedroot-causeanalysisofpower- powerdevicewithreverseconductionaccelerated powercyclingtest,”
converter failure in wind turbines,” IEEE Trans. Power Electron., inProc.IEEEAppl.PowerElectron.Conf.Expo.(APEC),Mar.2017,
vol.30,no.5,pp.2481–2492,May2015. pp.756–760.
[21] P.Drexhage,“Effectofhumidityandcondensationonpowerelectron- [43] S. Song, S. Munk-Nielsen, and C. Uhrenfeldt, “Failure mechanism
icssystems,”Semikron,Nuremberg,Germany,Appl.NoteAN16-001, analysis of off-state drain-to-source leakage current failure of acom-
2016. mercial650vdiscreteGaN-on-SiHEMTpowerdevicebyaccelerated
[22] H. Wang, D. A. Nielsen, and F. Blaabjerg, “Degradation testing and power cycling test,” Microelectron. Rel., vols. 76–77, pp.539–543,
failureanalysisofDCfilmcapacitorsunderhighhumidityconditions,” Sep.2017.
Microelectron. Rel.,vol.55,nos.9–10,pp.2007–2011,Aug.2015. [44] S.Song,S.Munk-Nielsen,andC.Uhrenfeldt,“Howcanacutting-edge
[23] T. Aichinger and M. Schmidt, “Gate-oxide reliability and failure-rate galliumnitridehigh-electron-mobility transistorencountercatastrophic
reduction ofindustrial SiCMOSFETs,”in Proc.IEEEInt.Rel.Phys. failure within the acceptable temperature range?” IEEETrans. Power
Symp.(IRPS),Apr.2020,pp.1–6. Electron.,vol.35,no.7,pp.6711–6718,Jul.2020.
[24] R. Singh, “Reliability and performance limitations in SiC power [45] J.Franke,G.Zeng,T.Winkler,andJ.Lutz,“Powercyclingreliability
devices,” Microelectron. Rel., vol. 46, nos. 5–6, pp.713–730, resultsofGaNHEMTdevices,” inProc.IEEE30thInt.Symp.Power
May2006. Semiconductor Devices ICs(ISPSD),May2018,pp.467–470.

6492 IEEEJOURNALOFEMERGINGANDSELECTEDTOPICSINPOWERELECTRONICS,VOL.9,NO.6,DECEMBER2021
[46] C. Abbate, G. Busatto, A. Sanseverino, D. Tedesco, and F. Velardi, [70] W.Lai,M.Chen,L.Ran,O.Alatise,S.Xu,andP.Mawby,“Lowstress
“Experimental study of the instabilities observed in 650 V enhance- cycleeffectinIGBTpowermoduledie-attachlifetimemodeling,”IEEE
ment mode GaN HEMT during short circuit,” Microelectron. Rel., Trans.PowerElectron.,vol.31,no.9,pp.6575–6585,Sep.2016.
vols.76–77,pp.314–320,Sep.2017. [71] W. Lai et al., “Experimental investigation on the effects of narrow
[47] M. Fernandez et al., “Short-circuit study in medium-voltage GaN junction temperature cycles on die-attach solder layer in an IGBT
cascodes, p-GaN HEMTs, and GaN MISHEMTs,” IEEE Trans. Ind. module,”IEEETrans.PowerElectron.,vol.32,no.2,pp.1431–1441,
| Electron.,      | vol.64,no.11,pp.9012–9022,Nov.2017. |          |                 |     |             |     |                 | Feb.2017.          |     |                    |     |     |             |                  |     |
| --------------- | ----------------------------------- | -------- | --------------- | --- | ----------- | --- | --------------- | ------------------ | --- | ------------------ | --- | --- | ----------- | ---------------- | --- |
|                 |                                     |          |                 |     |             |     |                 | [72] Semiconductor |     | Devices–Mechanical |     |     | andClimatic | TestMethods—Part |     |
| [48] C. Abbate, | G.                                  | Busatto, | A. Sanseverino, |     | D. Tedesco, |     | and F. Velardi, |                    |     |                    |     |     |             |                  |     |
“Failureanalysisof650VenhancementmodeGaNHEMTaftershort 34—PowerCycling, document IEC60749-34:2011, 2011.
circuittests,”Microelectron.Rel.,vols.88–90,pp.677–683,Sep.2018. [73] U. Scheuermann and S. Schuler, “Power cycling results for dif-
[49] C. Abbate, G. Busatto, A. Sanseverino, D. Tedesco, and F. Velardi, ferent control strategies,” Microelectron. Rel., vol. 50, nos. 9–11,
“FailuremechanismsofenhancementmodeGaNpowerHEMTsoper- pp.1203–1209,Sep.2010.
[74] M.A.Miner,“Cumulativedamageinfatigue,”J.Appl.Mech.,vol.12,
| ated | in short | circuit,” | Microelectron. | Rel., | vols. | 100–101, | Sep. 2019, |     |     |     |     |     |     |     |     |
| ---- | -------- | --------- | -------------- | ----- | ----- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
no.3,pp.159–164,1945.
Art.no.113454.
[50] X.Lyuetal.,“Areliableultra-fast shortcircuitprotection methodfor [75] A.FatemiandL.Yang,“Cumulativefatiguedamageandlifeprediction
E-mode GaN HEMT,” IEEE Trans. Power Electron., vol. 35, no. 9, theories: Asurveyofthestateoftheartforhomogeneous materials,”
pp.8926–8933,Sep.2020. Int.J.Fatigue, vol.20,no.1,pp.9–34,Jan.1998.
|               |         |          |                    |        |          |                     |             | [76] P. Rajaguru, |        | H. Lu,  | and C.     | Bailey, “Application |          | of nonlinear | fatigue |
| ------------- | ------- | -------- | ------------------ | ------ | -------- | ------------------- | ----------- | ----------------- | ------ | ------- | ---------- | -------------------- | -------- | ------------ | ------- |
| [51] Z. Shen, | Q.      | Wang, Y. | Shen,              | and H. | Wang,    | “First observations | in          |                   |        |         |            |                      |          |              |         |
|               |         |          |                    |        |          |                     |             | damage            | models | inpower | electronic | module               | wirebond | structure    | under   |
| degradation   | testing | of       | planar magnetics,” |        | in Proc. | IEEE                | Appl. Power |                   |        |         |            |                      |          |              |         |
Electron. Conf.Expo.(APEC),Mar.2019,pp.1436–1443. various amplitude loadings,” Adv. Manuf., vol. 2,no. 3, pp.239–250,
| [52] Z. Shen, | Q. Wang, | and | H. Wang, | “Degradation |     | analysis | of planar | Mar.2014. |          |         |                |     |             |     |                |
| ------------- | -------- | --- | -------- | ------------ | --- | -------- | --------- | --------- | -------- | ------- | -------------- | --- | ----------- | --- | -------------- |
|               |          |     |          |              |     |          |           | [77] G.   | Zeng, C. | Herold, | T. Methfessel, |     | M. Schafer, | O.  | Schilling, and |
magnetics,”inProc.IEEEAppl.PowerElectron.Conf.Expo.(APEC),
J.Lutz,“Experimentalinvestigationoflinearcumulativedamagetheory
Mar.2020,pp.2687–2693.
withpowercyclingtest,”IEEETrans.PowerElectron.,vol.34,no.5,
[53] GuidelineAQG324,QualificationofPowerModulesforuseinPower
pp.4722–4728,May2019.
| Electronics | Converter |     | Units in | Motor | Vehicles, | ECPE, | Nuremberg, |     |     |     |     |     |     |     |     |
| ----------- | --------- | --- | -------- | ----- | --------- | ----- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
Germany,2019. [78] U.-M.Choi,K.Ma,andF.Blaabjerg,“Validationoflifetimeprediction
|     |     |     |     |     |     |     |     | ofIGBT | modules | basedon | linear | damage | accumulation |     | bymeans of |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------- | ------- | ------ | ------ | ------------ | --- | ---------- |
[54] P.O’ConnorandA.Kleyner,PracticalReliabilityEngineering.Hobo-
superimposedpowercyclingtests,”IEEETrans.Ind.Electron.,vol.65,
ken,NJ,USA:Wiley,2012.
no.4,pp.3520–3529,Apr.2018.
| [55] H. Wang, | P. Diaz | Reigosa, | and | F. Blaabjerg, | “A  | humidity-dependent |     |               |     |         |          |           |     |           |               |
| ------------- | ------- | -------- | --- | ------------- | --- | ------------------ | --- | ------------- | --- | ------- | -------- | --------- | --- | --------- | ------------- |
|               |         |          |     |               |     |                    |     | [79] H. Wang, | P.  | Davari, | H. Wang, | D. Kumar, | F.  | Zare, and | F. Blaabjerg, |
lifetimederatingfactorforDCfilmcapacitors,”inProc.IEEEEnergy
Convers. Congr.Expo.(ECCE),Sep.2015,pp.3064–3068. “Lifetime estimation ofDC-link capacitors inadjustable speed drives
|                 |                                        |     |     |     |     |     |           | undergridvoltageunbalances,” |     |     |     | IEEETrans.PowerElectron.,vol.34, |     |     |     |
| --------------- | -------------------------------------- | --- | --- | --- | --- | --- | --------- | ---------------------------- | --- | --- | --- | -------------------------------- | --- | --- | --- |
| [56] R.Bayerer, | T.Herrmann,T.Licht,J.Lutz,andM.Feller, |     |     |     |     |     | “Modelfor |                              |     |     |     |                                  |     |     |     |
no.5,pp.4064–4078,May2019.
| power      | cycling                      | lifetime | of IGBT | modules-various |     | factors | influencing |         |            |            |        |               |     |        |            |
| ---------- | ---------------------------- | -------- | ------- | --------------- | --- | ------- | ----------- | ------- | ---------- | ---------- | ------ | ------------- | --- | ------ | ---------- |
|            |                              |          |         |                 |     |         |             | [80] M. | Dbeiss and | Y. Avenas, | “Power | semiconductor |     | ageing | test bench |
| lifetime,” | inProc.CIPS,Mar.2008,pp.1–6. |          |         |                 |     |         |             |         |            |            |        |               |     |        |            |
dedicatedtophotovoltaicapplications,”IEEETrans.Ind.Appl.,vol.55,
[57] U.-M.Choi,F.Blaabjerg,andS.Jorgensen,“Studyoneffectofjunction
no.3,pp.3003–3010,May2019.
temperatureswingdurationonlifetimeoftransfermoldedpowerIGBT [81] M. Dbeiss, Y. Avenas, H. Zara, L. Dupont, and F. A. Shakarchi,
modules,”IEEETrans.PowerElectron.,vol.32,no.8,pp.6434–6443,
|     |     |     |     |     |     |     |     | “A  | method for | accelerated | aging | tests | of power | modules | for photo- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | ----- | ----- | -------- | ------- | ---------- |
Aug.2017.
voltaicinvertersconsideringtheinvertermissionprofiles,”IEEETrans.
| [58] R.SchmidtandU.Scheuermann, |     |     |     | “Separating |     | failure modesinpower |     |                 |     |                                       |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | ----------- | --- | -------------------- | --- | --------------- | --- | ------------------------------------- | --- | --- | --- | --- | --- |
|                                 |     |     |     |             |     |                      |     | PowerElectron., |     | vol.34,no.12,pp.12226–12234,Dec.2019. |     |     |     |     |     |
cyclingtests,”inProc.Int.Conf.Integr.PowerElectron.Syst.(CIPS), [82] A.Sangwongwanich etal., “Mission profile-based accelerated testing
Mar.2012,pp.1–6. of DC-link capacitors in photovoltaic inverters,” in Proc. IEEE Appl.
[59] M. Junghaenel, R. Schmidt, J. Strobel, and U. Scheuermann, “Inves- PowerElectron. Conf.Expo.(APEC),Mar.2019,pp.2833–2840.
| tigationonisolatedfailuremechanismsinactive |     |     |     |     |     | powercycletesting,” |     |                  |     |            |     |        |          |         |              |
| ------------------------------------------- | --- | --- | --- | --- | --- | ------------------- | --- | ---------------- | --- | ---------- | --- | ------ | -------- | ------- | ------------ |
|                                             |     |     |     |     |     |                     |     | [83] I. Vernica, | F.  | Blaabjerg, | and | K. Ma, | “Mission | profile | emulator for |
inProc.PCIMEur.,May2015,pp.251–258.
|     |     |     |     |     |     |     |     | the | power electronics |     | systems | of motor | drive | applications,” | in Proc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------- | -------- | ----- | -------------- | -------- |
[60] N. Chemi-Con, “Judicious use of Aluminum electrolytic capacitor,” 19th Eur.Conf. Power Electron. Appl. (EPEECCE Eur.),Sep. 2017,
| Nippon | Chemi-Con, | Tokyo, | Japan, | Tech. | Note, | CAT. | No. E1001-L, | pp.1–10. |     |     |     |     |     |     |     |
| ------ | ---------- | ------ | ------ | ----- | ----- | ---- | ------------ | -------- | --- | --- | --- | --- | --- | --- | --- |
2011. [84] K.MaandY.Song,“Power-electronic-basedelectricmachineemulator
[61] U.ScheuermannandR.Schmidt,“Anewlifetimemodelforadvanced
|       |         |               |       |     |           |         |            | using | direct | impedance | regulation,” |     | IEEE Trans. | Power | Electron., |
| ----- | ------- | ------------- | ----- | --- | --------- | ------- | ---------- | ----- | ------ | --------- | ------------ | --- | ----------- | ----- | ---------- |
| power | modules | with sintered | chips | and | optimized | al wire | bonds,” in |       |        |           |              |     |             |       |            |
vol.35,no.10,pp.10673–10680,Oct.2020.
Proc.PCIMEur.,May2013,pp.810–813. [85] Z. Wang, H. Wang, Y. Zhang, and F. Blaabjerg, “A viable mission
[62] U. Scheuermann and M. Junghaenel, “Limitation of power module profileemulatorforpowermodulesinmodularmultilevel converters,”
lifetime derived from active power cycling tests,” in Proc. Int. Conf. IEEE Trans. Power Electron., vol. 34, no. 12, pp.11580–11593,
| Integr. | PowerElectron.Syst.(CIPS),Mar.2018,pp.1–10. |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ------- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Dec.2019.
[63] U. Scheuermann and U. Hecht, “Power cycling lifetime of advanced [86] Y.Shen,A.Chub,H.Wang,D.Vinnikov,E.Liivik,andF.Blaabjerg,
powermodulesfordifferenttemperatureswings,”inProc.PCIMEur., “Wear-out failure analysis of an impedance-source PV microinverter
vol.5964,May2002,pp.1–6. based on system-level electrothermal modeling,” IEEE Trans. Ind.
[64] D.-P. Sadik, J.-K. Lim, F. Giezendanner, P. Ranstad, and H.-P. Nee, Electron.,vol.66,no.5,pp.3914–3927,May2019.
“Humidity testing of SiC power MOSFETs—An update,” in Proc. [87] Y. Zhang, H. Wang, Z. Wang, F. Blaabjerg, and M. Saeedifard,
19th Eur.Conf. Power Electron. Appl. (EPEECCE Eur.),Sep. 2017, “Mission profile-based system-level reliability prediction method for
pp.1–8. modularmultilevel converters,” IEEETrans.PowerElectron.,vol.35,
[65] S. Kremp and O. Schilling, “Humidity robustness for high voltage no.7,pp.6916–6930,Jul.2020.
power modules: Limiting mechanisms and improvement of lifetime,” [88] I. Vernica, H. Wang, and F. Blaabjerg, “Design for reliability and
Microelectron. Rel.,vols.88–90,pp.447–452,Sep.2018. robustnesstoolplatform forpowerelectronic systems—Studycaseon
|                  |     |        |              |      |      |            |          | motordrive | applications,” |     | inProc.IEEEAppl.PowerElectron. |     |     |     | Conf. |
| ---------------- | --- | ------ | ------------ | ---- | ---- | ---------- | -------- | ---------- | -------------- | --- | ------------------------------ | --- | --- | --- | ----- |
| [66] S. Hartmann |     | and E. | Özkol, “Bond | wire | life | time model | based on |            |                |     |                                |     |     |     |       |
temperaturedependentyieldstrength,”inProc.PCIMEur.,May2012, Expo.(APEC),Mar.2018,pp.1799–1806.
pp.494–501. [89] Standard Practices for Cycle Counting in Fatigue Analysis, Stan-
[67] G. Zeng, R. Alvarez, C. Kunzel, and J. Lutz, “Power cycling results dard E1049-85, ASTM International, West Conshohocken, PA, USA,
| ofhighpowerIGBTmodulescloseto50hzheatingprocess,”inProc. |            |       |           |       |           |        |            | 2017.      |            |     |          |               |     |                  |     |
| -------------------------------------------------------- | ---------- | ----- | --------- | ----- | --------- | ------ | ---------- | ---------- | ---------- | --- | -------- | ------------- | --- | ---------------- | --- |
|                                                          |            |       |           |       |           |        |            | [90] S. K. | Chaudhary, | P.  | Ghimire, | F. Blaabjerg, |     | P. B. Thogersen, | and |
| 21st                                                     | Eur. Conf. | Power | Electron. | Appl. | (EPE ECCE | Eur.), | Sep. 2019, |            |            |     |          |               |     |                  |     |
pp.1–10. P.dePlace Rimmen,“Development offielddata logger forrecording
[68] U. Scheuermann and R. Schmidt, “Impact of load pulse duration on mission profile of power converters,” in Proc. 17th Eur.Conf. Power
powercyclinglifetimeofalwirebonds,”Microelectron. Rel.,vol.53, Electron. Appl.(EPEECCE-Eur.),Sep.2015,pp.1–10.
|     |     |     |     |     |     |     |     | [91] Y. Yang, | H.  | Wang, F. | Blaabjerg, | and | K. Ma, | “Mission | profile based |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | ---------- | --- | ------ | -------- | ------------- |
nos.9–11,pp.1687–1691,Sep.2013.
|                    |     |        |              |         |     |         |                | multi-disciplinary |     | analysis | of  | power | modules | in single-phase | trans- |
| ------------------ | --- | ------ | ------------ | ------- | --- | ------- | -------------- | ------------------ | --- | -------- | --- | ----- | ------- | --------------- | ------ |
| [69] M. Junghaenel |     | and U. | Scheuermann, | “Impact |     | of load | pulse duration |                    |     |          |     |       |         |                 |        |
on power cycling lifetime of chip interconnection solder joints,” formerless photovoltaic inverters,” in Proc. 15th Eur. Conf. Power
Microelectron. Rel.,vols.76–77,pp.480–484,Sep.2017. Electron. Appl.(EPE),Sep.2013,pp.1–10.

WANGANDBLAABJERG:POWERELECTRONICSRELIABILITY:STATEOFTHEARTANDOUTLOOK 6493
[92] Y.Zhang,H.Wang,Z.Wang,Y.Yang,andF.Blaabjerg, “Theimpact [113] C.Matei,J.Urbonas,H.Votsi,D.Kendig,andP.H.Aaen,“Dynamic
ofmissionprofile models onthe predicted lifetime ofIGBTmodules temperature measurements of a GaN DC–DC boost converter at
in the modular multilevel converter,” in Proc. Annu. Conf. IEEEInd. MHz frequencies,” IEEE Trans. Power Electron., vol. 35, no. 8,
Electron. Soc.(IECON),Oct.2017,pp.7980–7985. pp.8303–8310,Aug.2020.
[93] K.Ma,N.He,M.Liserre,andF.Blaabjerg, “Frequency-domain ther- [114] Y. Avenas, L. Dupont, and Z. Khatir, “Temperature measure-
mal modeling and characterization of power semiconductor devices,” ment of power semiconductor devices by thermo-sensitive electrical
IEEE Trans. Power Electron., vol. 31, no. 10, pp.7183–7193, parameters—A review,” IEEE Trans. Power Electron., vol. 27, no. 6,
| Oct.2016. |     |     |     |     |     |     |     | pp.3081–3092,Jun.2012. |     |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- |
[94] Y. Zhang, H.Wang,Z.Wang,Y.Yang, andF.Blaabjerg, “Simplified [115] S.Zhao,S.Chen,andH.Wang,“Degradationmodelingforreliability
thermalmodelingforIGBTmoduleswithperiodicpowerlossprofiles estimation of DC film capacitors subject to humidity acceleration,”
inmodularmultilevelconverters,”IEEETrans.Ind.Electron.,vol.66, Microelectron. Rel.,vols.100–101, Sep.2019,Art.no.113401.
no.3,pp.2323–2332,Mar.2019. [116] K. A. Severson et al., “Data-driven prediction of battery cycle
[95] H. Wang, R. Zhu, H. Wang, M. Liserre, and F. Blaabjerg, “A life before capacity degradation,” Nature Energy, vol. 4, no. 5,
thermal modeling method considering ambient temperature dynam- pp.383–391,Mar.2019.
ics,” IEEE Trans. Power Electron., vol. 35, no. 1, pp.6–9, [117] P.M.Attiaetal.,“Closed-loopoptimizationoffast-chargingprotocols
Jan.2020. for batteries with machine learning,” Nature, vol. 578, no. 7795,
[96] G.Mandrusiak,X.She,A.M.Waddell,andS.Acharya,“Onthetran- pp.397–402,Feb.2020.
| sientthermalcharacteristics |      |              | ofsiliconcarbidepowerelectronics |      |         |                   | mod- |     |     |     |     |     |     |     |     |
| --------------------------- | ---- | ------------ | -------------------------------- | ---- | ------- | ----------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
| ules,”                      | IEEE | Trans. Power | Electron.,                       | vol. | 33, no. | 11, pp.9783–9789, |      |     |     |     |     |     |     |     |     |
Nov.2018.
|         |           |                          |     |     |             |     |              |     |     | Huai | Wang (Senior | Member,       | IEEE)       | received | the      |
| ------- | --------- | ------------------------ | --- | --- | ----------- | --- | ------------ | --- | --- | ---- | ------------ | ------------- | ----------- | -------- | -------- |
| [97] A. | Tsibizov, | I. Kovacevic-Badstubner, |     |     | B. Kakarla, | and | U. Grossner, |     |     |      |              |               |             |          |          |
|         |           |                          |     |     |             |     |              |     |     | B.E. | degree       | in electrical | engineering |          | from the |
“Accurate temperature estimation of SiC power MOSFETs under Huazhong University of Science and Technology,
| extreme | operating | conditions,” |     | IEEETrans. | Power | Electron., | vol. 35, |     |     |        |        |          |     |           |        |
| ------- | --------- | ------------ | --- | ---------- | ----- | ---------- | -------- | --- | --- | ------ | ------ | -------- | --- | --------- | ------ |
|         |           |              |     |            |       |            |          |     |     | Wuhan, | China, | in 2007, | and | the Ph.D. | degree |
no.2,pp.1855–1865,Feb.2020.
|         |          |          |            |     |           |          |          |     |     | in power | electronics | from | the City | University | of  |
| ------- | -------- | -------- | ---------- | --- | --------- | -------- | -------- | --- | --- | -------- | ----------- | ---- | -------- | ---------- | --- |
| [98] Y. | Shen, H. | Wang, F. | Blaabjerg, | H.  | Zhao, and | T. Long, | “Thermal |     |     |          |             |      |          |            |     |
HongKong,HongKong,in2012.
modelinganddesignoptimizationofPCBviasandpads,”IEEETrans.
PowerElectron.,vol.35,no.1,pp.882–900,Jan.2020. He was a Visiting Scientist with ETH Zurich,
|         |            |         |              |     |                     |     |           |     |     | Zürich,   | Switzerland, | from              | August | to        | Septem- |
| ------- | ---------- | ------- | ------------ | --- | ------------------- | --- | --------- | --- | --- | --------- | ------------ | ----------------- | ------ | --------- | ------- |
| [99] U. | Schilling, | “Cosmic | ray failures | in  | power electronics,” |     | Semikron, |     |     |           |              |                   |        |           |         |
|         |            |         |              |     |                     |     |           |     |     | ber 2014, | and          | the Massachusetts |        | Institute | of      |
Nuremberg,Germany,Appl.NoteAN17-003,2017.
|          |          |              |     |           |            |     |               |     |     | Technology | (MIT), | Cambridge, |     | MA, USA, | from |
| -------- | -------- | ------------ | --- | --------- | ---------- | --- | ------------- | --- | --- | ---------- | ------ | ---------- | --- | -------- | ---- |
| [100] R. | Bayerer, | M. Lassmann, | and | S. Kremp, | “Transient |     | hygrothermal- |     |     |            |        |            |     |          |      |
SeptembertoNovember2013.HewaswiththeABB
responseofpowermodulesininverters—The basisformissionprofil- Corporate Research Center, Baden, Switzerland, in 2009. He is currently a
| ing | under climate | and | power loading,” |     | IEEE Trans. | Power | Electron., |           |          |                    |     |                   |          |     |         |
| --- | ------------- | --- | --------------- | --- | ----------- | ----- | ---------- | --------- | -------- | ------------------ | --- | ----------------- | -------- | --- | ------- |
|     |               |     |                 |     |             |       |            | Professor | with the | Center of Reliable |     | Power Electronics | (CORPE), |     | Depart- |
vol.31,no.1,pp.613–620,Jan.2016.
|                   |     |         |           |     |              |                   |     | ment of            | Energy Technology, |                 | Aalborg    | University, | Aalborg, | Denmark.       | His |
| ----------------- | --- | ------- | --------- | --- | ------------ | ----------------- | --- | ------------------ | ------------------ | --------------- | ---------- | ----------- | -------- | -------------- | --- |
| [101] S.Peyghami, |     | H.Wang, | P.Davari, | and | F.Blaabjerg, | “Mission-profile- |     |                    |                    |                 |            |             |          |                |     |
|                   |     |         |           |     |              |                   |     | research addresses |                    | the fundamental | challenges | in          | modeling | and validation |     |
basedsystem-levelreliabilityanalysisinDCmicrogrids,”IEEETrans.
|     |     |     |     |     |     |     |     | of power | electronic | component | failure | mechanisms | and | application | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --------- | ------- | ---------- | --- | ----------- | ------ |
Ind.Appl.,vol.55,no.5,pp.5055–5067,Sep.2019. in system-level predictability, condition monitoring, circuit architecture, and
| [102] S. | V. Dhople, | A.  | Davoudi, | A.  | D. Domínguez-García, |     | and |     |     |     |     |     |     |     |     |
| -------- | ---------- | --- | -------- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
robustnessdesign.
| P.L.Chapman,“A |     | unified | approach | to  | reliability | assessment | of mul- |     |     |     |     |     |     |     |     |
| -------------- | --- | ------- | -------- | --- | ----------- | ---------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
Dr.WangreceivedtheRichardM.BassOutstandingYoungPowerElectron-
| tiphase | DC–DC | converters   | in         | photovoltaic | energy   | conversion | sys-        |              |       |               |       |             |         |     |          |
| ------- | ----- | ------------ | ---------- | ------------ | -------- | ---------- | ----------- | ------------ | ----- | ------------- | ----- | ----------- | ------- | --- | -------- |
|         |       |              |            |              |          |            |             | ics Engineer | Award | from the IEEE | Power | Electronics | Society | in  | 2016 and |
| tems,”  | IEEE  | Trans. Power | Electron., |              | vol. 27, | no. 2,     | pp.739–751, |              |       |               |       |             |         |     |          |
theGreenTalentsAwardfromtheGermanFederalMinistryofEducationand
Feb.2012. Researchin2014.HeisalsotheChairoftheIEEEPELS/IAS/IESChapterin
| [103] J. | Pippola, | I. Vaalasranta, | T.  | Marttila, | J. Kiilunen, |     | and L. Frisk, |          |         |              |           |        |         |                 |     |
| -------- | -------- | --------------- | --- | --------- | ------------ | --- | ------------- | -------- | ------- | ------------ | --------- | ------ | ------- | --------------- | --- |
|          |          |                 |     |           |              |     |               | Denmark. | He also | serves as an | Associate | Editor | for the | IET Electronics |     |
“Productlevelacceleratedreliabilitytestingofmotordriveswithinput
|       |                 |     |             |       |            |      |         | Letters, IEEE       | JOURNALOFEMERGINGANDSELECTEDTOPICSINPOWER |                                 |     |     |     |     |     |
| ----- | --------------- | --- | ----------- | ----- | ---------- | ---- | ------- | ------------------- | ----------------------------------------- | ------------------------------- | --- | --- | --- | --- | --- |
| power | interruptions,” |     | IEEE Trans. | Power | Electron., | vol. | 30, no. | 5,                  |                                           |                                 |     |     |     |     |     |
|       |                 |     |             |       |            |      |         | ELECTRONICS,andIEEE |                                           | TRANSACTIONSONPOWERELECTRONICS. |     |     |     |     |     |
pp.2614–2622,May2015.
| [104] J. | Flicker, | G. Tamizhmani, | M.  | K.  | Moorthy, | R. Thiagarajan, | and |     |     |     |     |     |     |     |     |
| -------- | -------- | -------------- | --- | --- | -------- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
R.Ayyanar,“Acceleratedtestingofmodule-levelpowerelectronicsfor
long-term reliability,” IEEE J. Photovolt., vol. 7, no. 1, pp.259–267, FredeBlaabjerg(Fellow,IEEE)receivedthePh.D.
| Jan.2017. |     |     |     |     |     |     |     |     |     | degree | in electrical | engineering | from | Aalborg | Uni- |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------------- | ----------- | ---- | ------- | ---- |
versity,Aalborg,Denmark,in1995.
| [105] T. | Krone, L. | Dang Hung, | M.  | Jung, | and A. | Mertens, | “Advanced |     |     |     |     |     |     |     |     |
| -------- | --------- | ---------- | --- | ----- | ------ | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
condition monitoring system based on on-line semiconductor loss He was with ABB-Scandia, Randers, Denmark,
measurements,”inProc.IEEEEnergyConvers.Congr.Expo.(ECCE), from 1987 to 1988. He became an Assistant Pro-
Sep.2016,pp.1–8. fessor in 1992, an Associate Professor in 1996,
[106] B. Rannestad, A. E. Maarbjerg, K. Frederiksen, S. Munk-Nielsen, and a Full Professor of power electronics and
|     |              |            |     |            |      |              |         |     |     | drives | in 1998 | at Aalborg | University. | Since | 2017, |
| --- | ------------ | ---------- | --- | ---------- | ---- | ------------ | ------- | --- | --- | ------ | ------- | ---------- | ----------- | ----- | ----- |
| and | K. Gadgaard, | “Converter |     | monitoring | unit | for retrofit | of wind |     |     |        |         |            |             |       |       |
power converters,” IEEE Trans. Power Electron., vol. 33, no. 5, hehasbeenaVillumInvestigator. Heisthehonoris
pp.4342–4351,May2018. causaatUniversitatea Politehnica Timisoara(UPT),
[107] B. Rannestad, K. Fischer, P. Nielsen, K. Gadgaard, and Timisoara, Romania, and the Tallinn University of
Technology(TTU),Tallinn,Estonia.Hehaspublishedmorethan600journal
S.Munk-Nielsen,“Virtualtemperaturedetectionofsemiconductorsin
amegawattfieldconverter,”IEEETrans.Ind.Electron.,vol.67,no.2, articlesinthefieldsofpowerelectronicsanditsapplications.Heisacoauthor
pp.1305–1315,Feb.2020. offour monographs and aneditor of ten books in power electronics and its
[108] H. Wang and Y. Peng, “Non-invasive front-end for power electronic applications. His current research interests include power electronics and its
monitoring,” FiledPatent, PA202070235,Jul.21,2020. applications, suchasinwindturbines, photovoltaic (PV)systems,reliability,
|               |          |     |          |              |     |            |           | harmonics,andadjustable |     | speeddrives. |     |     |     |     |     |
| ------------- | -------- | --- | -------- | ------------ | --- | ---------- | --------- | ----------------------- | --- | ------------ | --- | --- | --- | --- | --- |
| [109] Z.Wang, | Y.Zhang, | H.  | Wang,and | F.Blaabjerg, |     | “Capacitor | condition |                         |     |              |     |     |     |     |     |
monitoring based on the DC-side start-up of modular multilevel con- Dr. Blaabjerg received 32 IEEE Prize Paper Awards, the IEEE PELS
verters,” IEEETrans.PowerElectron.,vol.35,no.6,pp.5589–5593, DistinguishedServiceAwardin2009,theEPE-PEMCCouncilAwardin2010,
Jun.2020. the IEEE William E. Newell Power Electronics Award in 2014, the Villum
[110] Y. Peng, S. Zhao, and H. Wang, “A digital twin based estimation KannRasmussenResearchAwardin2014,theGlobalEnergyPrizein2019,
|     |     |     |     |     |     |     |     | and the 2020 | IEEE | Edison Medal. | He  | was nominated |     | in 2014–2019 | by  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ------------- | --- | ------------- | --- | ------------ | --- |
methodforhealthindicatorsofDC–DCconverters,”IEEETrans.Power
Electron., vol.36,no.2,pp.2105–2118,Feb.2021. ThomsonReuterstobeamongthe250most-citedresearchers inengineering
[111] S.Zhao,S.Chen, F.Yang,E.Ugur,B.Akin,andH.Wang,“Acom- in the world. He was the Editor-in-Chief of the IEEE TRANSACTIONSON
positefailureprecursorforconditionmonitoringandremaininguseful POWERELECTRONICSfrom2006to2012.HewasaDistinguishedLecturer
lifeprediction ofdiscretepowerdevices,”IEEETrans.Ind.Informat., of the IEEE Power Electronics Society from 2005 to 2007 and the IEEE
vol.17no.1,pp.688–698,Jan.2021. Industry Applications Society from 2010 to 2011 and 2017 to 2018. From
[112] S. Zhao, F. Blaabjerg, and H. Wang, “An overview of artificial intel- 2019to2020,hehasservedasthePresidentoftheIEEEPowerElectronics
ligence applications for power electronics,” IEEETrans. Power Elec- Society. He is also the Vice-President of the Danish Academy of Technical
| tron.,earlyaccess,Sep.18,2020,doi:10.1109/TPEL.2020.3024914. |     |     |     |     |     |     |     | Sciences. |     |     |     |     |     |     |     |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |