Received2January2023,accepted2February2023,dateofpublication6February2023,dateofcurrentversion13February2023.
DigitalObjectIdentifier10.1109/ACCESS.2023.3242918
Fault Management Techniques to Enhance
the Reliability of Power Electronic
Converters: An Overview
SAEEDRAHIMPOUR 1,(StudentMember,IEEE),
OLEKSANDRHUSEV 1,(SeniorMember,IEEE),DMITRIVINNIKOV 1,(Fellow,IEEE),
NASERVOSOUGHIKURDKANDI 2,(Member,IEEE),AND
HADITARZAMNI 3,(StudentMember,IEEE)
1DepartmentofElectricalPowerEngineeringandMechatronics,TallinnUniversityofTechnology,19086Tallinn,Estonia
2DepartmentofElectricalandComputerEngineering,SanDiegoStateUniversity,SanDiego,CA92182,USA
3DepartmentofElectricalEngineeringandAutomation,AaltoUniversity,02150Espoo,Finland
Correspondingauthor:HadiTarzamni(hadi.tarzamni@aalto.fi)
ThisworkwassupportedinpartbytheEstonianResearchCouncilunderGrantPRG675;inpartbytheEstonianCentreofExcellencein
ZeroEnergyandResourceEfficientSmartBuildingsandDistricts(ZEBE);inpartbytheJennyandAnttiWihuriFoundation;andinpart
bytheFinnishElectronicLibrary(FinElib),Finland,undertheFinELibconsortium’sagreementwiththeInstituteofElectricaland
ElectronicsEngineers(IEEE).
ABSTRACT The reliability of power electronic converters is a major concern in industrial applications
because of using prone-to-failure elements such as high-power semiconductor devices and electronic
capacitors. Hence, designing fault-tolerant inverters has been of great interest among researchers in both
academiaandindustryoverthelastdecade.Amongthethreestagesoffaultmanagement,compensatingthe
faultisthemostimportantandchallengingpart.Thetechniquesforfaultcompensationcanbeclassifiedinto
threegroups:hardwareredundancymethodswhichuseextraswitches,legs,ormodulestoreplacethefaulty
partsdirectlyorindirectly,switchingstatesredundancymethodswhichareaboutomittingandreplacingthe
impossible switching states, and unbalance compensation including the techniques to compensate for the
unbalancesinthesystemcausedbyafault.Inthispaper,anoverviewoffault-tolerantinvertersispresented.
A classification of fault-tolerant inverters is demonstrated and major cases in each of its categories are
explained.
INDEX TERMS Reliability, fault compensation, fault-tolerant converter, fault management, multilevel
inverter,redundancy.
I. INTRODUCTION Reliability is defined as the ability of an item to perform
Inrecentdecades,powerelectronicshavebeenincreasingly a required function under stated conditions for a certain
used in modern power systems. Power electronic convert- period[6].Itisoftenmeasuredbytheprobabilityoffailure,
ers are the main energy conversion system in a wide range frequencyoffailure,ortermsofavailability.Theessenceof
of applications such as renewable energy systems, energy reliability engineering is to prevent the creation of failures
storagesystems,smartandmicrogridtechnologies,dctrans- andfaults.Afaultinapowerelectronicsystemnotonlymay
mission and distribution systems, electric motor drives, and cause an unscheduled interruption, which is not tolerated,
powersupplies[1],[2],[3],[4].Thewidespreaduseofpower but may even lead to a disastrous accident [7], [8]. These
electronic converters in various industries has made their unplanned interruptions may cause significant safety con-
reliableperformanceatoppriority[5]. cerns and an increase in system operation costs as well [9].
It is therefore clear that the push toward ever-more reliable
The associate editor coordinating the review of this manuscript and powerelectronicproductsiscriticalforallindustriespower
approvingitforpublicationwasBoPu . networks in rural areas, critical medical equipment power
13432 ThisworkislicensedunderaCreativeCommonsAttribution4.0License.Formoreinformation,seehttps://creativecommons.org/licenses/by/4.0/ VOLUME11,2023

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
FIGURE2. Classificationoffailuresinpowerconverters.
FIGURE1. DistributionofunscheduledmaintenanceeventsofaPVplant.
is lifetime management, which is mainly concerned with
the wear-out issue of the components and devices of the
supplies,aircraft,andnavalpowersystems,satellitesystems system.Itconsistsofthreemajorsubcategories:lifetimeanal-
with unfeasible maintenance, and wind and solar farm with ysis,lifetimeprediction,andlifetimeextension.However,the
extensive and widely distributed parts [10]. Considerable focusofthissurveyisjustonthefaultmanagementaspectof
manufacturershavebeengettingagrowingawarenessofthe reliability, especially fault compensation and the survey on
protection efficiency and maintenance costs of power elec- lifetimemanagementofpowerelectronicconvertershasbeen
tronicdevices[4].Hence,thereliabilityofpowerelectronics discussedin[16].
isrecognizedasoneofthetopresearchtopics,theimportance Power semiconductors and capacitors are the most
ofwhichisgrowingrapidly[11]. vulnerable power electronic components [18]; therefore,
Invertersplayanimportantroleinthereliabilityofelectri- the reliability of power converters mostly focuses on these
calsystemssuchasrenewableenergysystems,motordrives, failure-prone power electronic components. capacitors are
electric vehicles, etc. Industrial experiences show drives, sensitivetothermalandelectricalstressesandhavethemain
electric vehicles, etc. Industrial experiences show that con- disadvantage of low lifespan and high degradation failure
vertersarefrequentfailuresourcesinmanyapplicationssuch rate [19]. It is demonstrated in Fig. 4 that about 18% of
as wind and PV systems [12]. As an example, as shown the faults in converters are caused by the degradation of
in Fig. 1, inverters are responsible for about 37 percent of capacitors. Electrolytic capacitors, which are mostly used
unscheduled maintenance events in PV systems. Therefore, as dc-link capacitors, have the shortest lifetime among all
the reliability of inverters is a major concern in industrial capacitors. These capacitors are among the major failure
applications, especially due to the use of a large number of factors in PV inverters. Many efforts have been done to
high-powersemiconductordeviceswithhighpowerdensities improve the reliability of power electronic converters by
andhighfailurerates. minimizingdc-linkcapacitancesothatsmallcapacitorswith
Althoughtherearelotsofeffortstomaketwo-levelinvert- alonglifetimecanbeusedtoreplaceelectrolyticcapacitors.
ers reliable, the fault-tolerant ability is a more significant Hence, some research investigations have been devoted to
challengeinthemultilevelinverters,asthepossibilityoffail- reducingthesizeofcapacitorsininvertersaswellasreplacing
ureishigherfortheseconvertersduetothehighernumberof the electrolytic capacitors with non-electrolytic capacitors.
switchingdevices[13],[14],[15].Therefore,thestudyofthe However, these approaches include extra components along
faultmanagementoperationismainlyfocusedonmultilevel with the increased complexity of the switching patterns.
converters. In three-phase applications, a lower amount of capacitance
AsshowninFig.2,generally,thefailureinconverterscan canbeusedwherethepowerpulsationislower[20].Infact,
beclassifiedintocatastrophicfaultduetosingle-eventover- addressing the faults of capacitors is mostly focused on the
stressandwear-outfailureduetothelong-timedegradationof prevention of faults, and fault management techniques are
components.Fig.3demonstratesthegeneralguidelineforthe not discussed in the literature. however, due to the features
reliabilityofpowerelectronicconverters.Thisfiguredivides oftheswitches,therearemanymethodsthatcanameliorate
thediscussionofthereliabilityofpowerconvertersintotwo converters’faultconditions.
aspects: fault management and lifetime management. How- As was mentioned earlier, generally, the faults in power
ever,inresearch,Thereisusuallyadistinctionbetweenthese devices can be divided into two cases: SC fault and OC
two reliability areas [16]. Fault management is responsible fault. SC faults affecting the switches are the most serious
for managing the catastrophic faults in converters, such as faults [21]. An SC fault will produce an abnormal overcur-
Short-Circuit (SC) and Open-Circuit (OC) faults that can rent, causing serious damage to other parts within a very
causedestructivedamage[17].Theotheraspectofreliability shortperiodoftime[22].Therefore,SCfault-tolerantcontrol
VOLUME11,2023 13433

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
II. FAULTDIAGNOSIS
| Fault diagnosis | or fault         | detection |                 | is the         | first step    | once a   |
| --------------- | ---------------- | --------- | --------------- | -------------- | ------------- | -------- |
| fault occurs.   | Fault diagnostic |           | techniques      |                | for inverters | can      |
| be divided      | into model-based |           | and data-driven |                | methods.      | The      |
| model-based     | methods          | are based | on              | the analytical |               | model of |
theconverter[29].Theyusuallyneedtoconsiderthedynamic
| properties | and operation | mechanism    |     | of    | the system, | then  |
| ---------- | ------------- | ------------ | --- | ----- | ----------- | ----- |
| establish  | an accurate   | mathematical |     | model | [30]. The   | data- |
drivenfaultdiagnosismethodsdonotneedtoknowtheexact
| analytical        | model of the | system.     | they  | directly   | analyze | and        |
| ----------------- | ------------ | ----------- | ----- | ---------- | ------- | ---------- |
| process           | the measured | data [31].  | These | techniques |         | include    |
| signal processing | methods,     | statistical |       | analysis,  | and     | artificial |
FIGURE3. Guidelineofthereliabilityofpowerelectronicsystems[16].
intelligence.Thereisalsoahybridmethodthatusesacom-
binationofthesetwomethods.
Itisworthnotingthatthefaultdetectionmethodisnotthe
mainconcernofthispaper.In[32],faultDiagnosistechniques
| for Modular | Multilevel | Converters |           | (MMCs) | are reviewed. |           |
| ----------- | ---------- | ---------- | --------- | ------ | ------------- | --------- |
| References  | [33] and   | [34] have  | evaluated |        | IGBT’s        | different |
faultdiagnosisapproaches.References[35]and[36]present
acomprehensivesurveyoffaultdiagnosistechniques.
III. FAULTISOLATION
Faultisolationisthefirststepintacklingafaultinasystem.
Whenafaultoccursinaninverter,someswitchingstatesmay
| be unavailable | due to | the SC | or OC | of the | faulty | switches. |
| -------------- | ------ | ------ | ----- | ------ | ------ | --------- |
Theseswitchingstatesshouldbeavoidedorthefaultycom-
FIGURE4. Failuredistributionamongmajorcomponentsinatypical ponentsthemselvesshouldbeisolatedsothatthesystemcon-
converter. tinuestofunctionandpreventsdamagetothewholesystem.
Theseschemesareperformedbyaddingsomeextraelements
suchasfusesandTRIACsandtheirgoalistoisolatethefaulty
strategiesrelyheavilyonhardware[23].Thisfaultiscaused switch(es). Fault isolation usually results in the degradation
primarilybycontinuousgatepulses,overvoltage,aninternal ofthesystem’sperformance,especiallyintheoutputvoltage
fault caused by overheating, and freewheeling diode failure andTHD.Therefore,therehavetobesolutionstocompensate
caused by high reverse recovery voltages [24]. Fast fuse fortheeffectsofthefaultwhicharediscussedinsectionIII.
devices connected in series with power devices can convert In Fig. 5(a) [37] one phase of a three-phase two-level
shortcircuitfaultstoopencircuitfaultswheneverthefusible inverterisdemonstrated.IftheswitchS failsSC,first,the
a2
elementopens[25]. switchS a1 shouldbeturnedofftemporarily.Next,theTRIAC
A variety of mechanisms can cause OC faults, including T is turned on to make a shoot-through in the bottom dc
a
bond-wirelift-off,gatedriverfailure,orinternalconnection bus and blow the fuse F a2 . Now the S a2 is off the circuit.
rupture caused by thermal or mechanical shocks [26], [27]. Requiringaccesstothemidpointofthedc-linkandincreased
TheconverteroperatesatlowpowerqualityafteranOCfault, parasitic inductance because of the fuses are limitations of
causingadditionalstressesonitscircuitcomponents,causing thisapproach.
secondaryproblems[28]. InFig.5(b)[38],[39],whenS failsOC,S andT are
|     |     |     | a2  |     | a2  | a   |
| --- | --- | --- | --- | --- | --- | --- |
When a fault occurs, the fault management operation is turned on to create an SC across the top capacitor which
activatedwhichconsistsoffaultdiagnosisandfaultcompen- blowsthefuseF .Theinverterlegwhichcorrespondstothe
a
sation. phase ‘‘a’’ is now isolated. On the other hand, if S fails
a2
Thisworkfocusesonmethodstocompensateforthefaults. short, turning off S a1 is the first step. Then the TRIAC T a
Thesetechniquesoffaultcompensationininvertersareclas- is turned on which causes a shoot through in the bottom dc
sified.First,faultisolationtechniquesareexplainedwhichare buscapacitorandblowsFawhichisolatesthewholeleg‘‘a’’.
categorizedbyhardware-basedandmodulation-basedmeth- InFig.5(c)[40],whentheswitchS failseitherSCorOC,
a2
ods. Then the redundancy methods as the popular approach S is turned off and T is turned on. This creates a shoot-
| a1  |     | a1  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
toreconfiguringtheinverterinthepost-faultarediscussedin throughthatblowsthefuseF a2 whichremovestheswitchS a2
detail.Finally,thecontroltechniquestoregaintheconverter’s fromthecircuit.Requireahighnumberofcomponentsand
pre-faultperformancearedescribedusingfigures. increased parasitic inductance because of the fuses. are the
13434 VOLUME11,2023

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
drawbacksofthistechnique.Also,relativelylargecapacitors ensuringthenormaloperationoftheinverterwithtwohealthy
areneededtodecreasetheisolationtime. modules.Ifasecondfaultoccursintworemainingmodules,
In another approach which is shown in Fig. 5(d), after thefaultymodulewillbeeliminatedusingtherelatedrelays
isolating the faulty leg, the neutral point of the three-phase and the output voltage level will be decreased from 7 to
motorisforcedtoconnecttothedc-linkmidpointbyturning 3 so that the remaining modules can continue to operate.
ontheTRIACT . Themaindrawbacksofthementionedfault-tolerantscheme
a
One leg of a three-phase Neutral Point Clamped arethehighvoltagestressontheremainedhealthyswitches
(NPC) inverter with an isolation circuit is shown in anddecreasedoutputvoltagelevel.Therefore,theswitching
Fig. 5(e) [41], [42]. If S fails short, then the top dc-bus algorithmismodifiedtoallowtheinvertertoeithercontinue
a1
capacitorwillexperienceanSCthroughD a2 andF a2 during working in nominal condition, if it is possible or in derated
thezeroswitchingstateinwhichS andS areturnedon. operationalmode[53].
|                             |     |     | a2                      | a3  |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| Toavoidthisswitchingstate,T |     |     | isturnedontoblowthefuse |     |     |     |     |     |     |     |     |
a2
F a2 .Theinverteristurnedintoatwo-levelinverterwherethe IV. FAULTCOMPENSATION
outputvoltageisthesameasbefore,butitsTHDdecreases.
Afterfaultisolation,faultcompensationschemesarerequired
In the modified NPC inverter in [43], [44], and [45] to guarantee the operation of the faulty inverter as close
(Fig. 5(f)) the faulty phase is forced to connect to the dc- as possible to normal operation. In this paper, as shown in
| link midpoint | via | an additional | TRIAC. | After | faults, the |     |     |     |     |     |     |
| ------------- | --- | ------------- | ------ | ----- | ----------- | --- | --- | --- | --- | --- | --- |
Fig.9,faultcompensationtechniquesareclassifiedintothree
reconfigured system is similar to the structure where only groups: hardware redundancy, switching states redundancy,
fourswitchesareusedtodriveathree-phasemachine.Since
|     |     |     |     |     |     | and unbalance | compensation | control. | In  | order | to choose a |
| --- | --- | --- | --- | --- | --- | ------------- | ------------ | -------- | --- | ----- | ----------- |
theinverterisstillcapableofprovidingthefullratedcurrent, suitablefault-tolerantmethod,outputperformanceconsisting
themaximumbalancedline-to-lineoutputvoltageinpostfault of factors including the total harmonic distortion (THD) of
operationsisreducedtohalfofitsnominalvalue.Thelimita-
|     |     |     |     |     |     | output voltages | or currents, | system | efficiency, |     | and dynamic |
| --- | --- | --- | --- | --- | --- | --------------- | ------------ | ------ | ----------- | --- | ----------- |
tionsofthisapproacharerequiringaccesstothemidpointof responseshouldbeconsidered.Costisanotherimportantfac-
thedc-link,andoversizeddc-buscapacitors. torincomparingdifferenttechniquesoffaultcompensation.
| ANPC | converter | shown | in Fig. 5(g) | can be | operated as |            |            |      |           |     |             |
| ---- | --------- | ----- | ------------ | ------ | ----------- | ---------- | ---------- | ---- | --------- | --- | ----------- |
|      |           |       |              |        |             | Redundancy | means that | when | a feature | of  | a system is |
a three-level leg after a single-switch SC fault [41], [42]. down, it can be replaced by another feature that is already
| For example, | if  | S fails | short, thyristor | T is | turned on to |     |     |     |     |     |     |
| ------------ | --- | ------- | ---------------- | ---- | ------------ | --- | --- | --- | --- | --- | --- |
a1 a2 included in the system [54]. In the case of inverters, redun-
blow fuse F 2 . However, unlike NPC, the zero state still can dancy can be either switching state redundancy, which
be obtained by turning on switches S and S . The other includesalternativecurrentpathstoobtainthesamevoltage
|     |     |     | a2  | a5  |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
switchingstatesremainunchanged.Afterthefault,theoutput
level,orhardwareredundancy,whichincludesextraswitches,
voltage will experience no change in value, however, the legs,andmodules.
voltagestressonthehealthydevicesequalsdc-busvoltage.
The cascaded H-bridge (CHB) inverter shown in Fig. 6, A. HARDWAREREDUNDANCY
isoneofthepopularconvertertopologiesusedinhigh-power Redundant hardware techniques involve adding some
| medium-voltage |     | motor drives | to achieve | medium-voltage |     |           |                 |          |         |     |             |
| -------------- | --- | ------------ | ---------- | -------------- | --- | --------- | --------------- | -------- | ------- | --- | ----------- |
|                |     |              |            |                |     | redundant | hardware to the | original | system. | The | addition of |
withlowharmonicdistortion[46].ThewideadoptionofCas- hardware increases the cost of the system, but it provides
cadedMultilevelConverters(CMC)andModularMultilevel advantagesinpost-faultoperations,especiallyinapplications
Converter(MMC)inthehigh-voltagedirectcurrent(HVDC) wherecostisnotamajorconcern[8].
industry is mainly due to their modularity, scalability, and A simple solution for switch redundancy is shown in
| inherent | fault tolerance | [47], | [48], [49]. | It is composed | of  |     |     |     |     |     |     |
| -------- | --------------- | ----- | ----------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
Fig.10(a)whichhandlesSCoftheswitchesS1andS2[55],
a number of modular H-bridge power cells and isolated dc but it suffers from voltage sharing problems and doubled
voltagepowersources,whichcanbeobtainedfromthephase- conductionlossesinhealthyconditions.
shiftingtransformeranddioderectifiers[50]. The topology in Fig. 10(b) can resolve both OC and SC
The CMC topology (Fig. 7) has inherent module-level faults by using TRIACs [56]. It does not have the problems
| redundancy[47],[51].Ifamodulee.g.,A |     |     |     | ,experiencesfail- |     |     |     |     |     |     |     |
| ----------------------------------- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
1 of the previous topology; however, its cost is higher due to
ure,itisbypassedbytheTRIACT A1 andthecorresponding thenumberofcomponents.
healthymodulesoftwootherphaseswhichareB andC can A fault-tolerant switch-redundant flying capacitor leg is
|     |     |     |     |     | 1 1 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bebypassedformakingthevoltagebalanced[50].However, introducedin[57].whichisdemonstratedinFig.10(c).When
asdemonstratedinFig.7,thesymmetryoutputvoltagesare oneoftheswitchesfails,itscomplementaryswitchisturned
| achieved | with a | 33% amplitude | reduction, | which | limits the |                   |         |          |     |           |                |
| -------- | ------ | ------------- | ---------- | ----- | ---------- | ----------------- | ------- | -------- | --- | --------- | -------------- |
|          |        |               |            |       |            | on. The redundant | cell is | composed | of  | R 1 , R 2 | , C R replaces |
operationrangeunderfault. the faulty one. The FC leg continues to provide the same
In[52],thesuggestedstructure(Fig.8)hasfourrelaysin outputvoltage.Duringnormalmode,theadditionalcellisin
each module. The relays mentioned above are connected so apermanenton-state[57].
thatintheeventofanSCorOCfailure,thedefectivemod- Intheswitch-redundanttopologyinFig.11,ifoneofthe
| ule can be    | eliminated | and | isolated from | the whole | system, |                |          |        |        |             |        |
| ------------- | ---------- | --- | ------------- | --------- | ------- | -------------- | -------- | ------ | ------ | ----------- | ------ |
|               |            |     |               |           |         | upper switches | fails OC | or SC, | it can | be replaced | by the |
| VOLUME11,2023 |            |     |               |           |         |                |          |        |        |             | 13435  |

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
FIGURE5. Structureoffaultisolationapproaches.
FIGURE7. ThephasordiagramofCMCinverter:(a)normalmode,
(b)underfaultcondition.
FIGURE6. Fault-tolerantstructureofCHBinverter.
redundantswitchS bythecorrespondentrelay.Thestrategy suggestedby[59]inordertomaintaintheoutputvoltageof
R1
forthefailureofbottomswitchesisthesame. the inverter at the same value before the occurrence of the
The topology presented in [59] proposes a fault-tolerant fault.
five-level inverter for PV applications consisting of a Some hardware redundant topologies use the redundancy
two-level half-bridge inverter, a three-level diode-clamped of a whole leg to make the leg replicable when a probable
inverter, and a bidirectional switch made with four diodes. faultoccurs.Theredundantlegcanbeconnectedinparallel
Asaresultofaswitchfaultordc-sourcefault,thetopology orinseries.
operatesasathree-level,resultinginhalftheoutputvoltage. Theconverterin[60]isbasedonaback-to-backconverter
Twoadditionalswitchesandacenter-tappedtransformerare and S and S as its redundant switches. One leg of this
7 8
13436 VOLUME11,2023

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
FIGURE8. Thefault-tolerantinverterproposedin[52].
FIGURE12. Faulttoleranttopologiesusinglegredundancy.
FIGURE9. Classificationoffaultcompensationtechniquesforpower
converters.
FIGURE13. Fault-tolerantinverterproposedin[61].
ThetopologyinFig.12(b),doesnotusefusesandTRIAC.
Turningontherelayremovesthefaultylegandconnectsthe
redundantleg.However,itcannothandletwoSCswitchesin
oneleg.
In [32], as shown in Fig. 13, switches S and S act as
2 2
redundantswitches.Innormaloperation,duringthepositive
half cycle of the current i , the TRIACs T and S are
O 1 2
continuouslyon.Thepoweringodeisobtainedbyturningon
FIGURE10. Faulttoleranttopologiesusingseriesswitchredundancy.
S andS andthefreewheelingmodeisobtainedwheneither
1 4
S and D or D and S conduct. During the negative half-
1 2 3 4
cycle of i , the TRIACs S and S are on instead of S and
O 7 8 5
S .Inthisconfiguration,theswitchesS andS andthediodes
6 1 4
D andD areutilizedinthesamewayasinthepositivehalf
2 3
cycle.
When S fails short, S is turned on to obtain powering
1 4
mode,andS isturnedofftoobtainfreewheelingmode.Inthe
4
event of an OC fault in S or S , the converter continues
1 4
to function using S and S . In this case, while during the
2 3
negative half cycle the TRIACs S and S are on, S and
5 6 7
FIGURE11. Fault-tolerantinverterusingparallelswitchredundancy S 8 are on during the positive half cycle. During the normal
proposedin[58]. operation,S andS usethesameswitchingstrategyasS and
2 3 1
S .WhenS experiencesanOCfault,S isturnedoffperma-
4 7 8
topologyisshowninFig.12(a).Ifoneoftheswitchese.g.,S nently,andS andS arepermanentlyturnedon.Asaresult,
1 5 6
failsOC,theTRIACT isturnedontoconnectwhichmakes theconverterispermanentlyreconfiguredtotheconventional
1
thefaultyleggetreplacedwiththeredundantlegcontaining VSIconfiguration.AssoonasanSCfaultoccursinS ,S will
7 8
S and S . In the SC case, the faulty leg is isolated by very beturnedON,andS andS willbepermanentlyturnedoff.
7 8 5 6
fast-actingfuses;consequently,theSCfaultbecomesanOC and the circuit is permanently changed to the conventional
faultaftertheisolationofthefaultylegbythetwofuses. VSIconfiguration.
VOLUME11,2023 13437

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
FIGURE14. Fault-tolerantleg-redundantinvertersproposedin[62].
In [62] a fourth leg is added to the conventional NPC components count, preserving high efficiency in postfault
inverter which is connected to the neutral point of the con- operation,andavoidingtheusageofbidirectionalswitches.
verter through an inductance. This fourth leg apart from its In the topology in Fig. 15(b) proposed in [55], the main
duty as a redundant leg for the postfault operation works invertercomprisesaconventionalthree-levelNPCleganda
under the normal operation as well to balance the Neutral conventionalthree-levelFClegalongwitharedundantbridge
pointvoltagebyinjectingthelocallyaveragedcurrenttothe attheoutputterminalastheredundantleg.
neutralpoint.Relaysareaddedtoreconfiguretheconverteras WhenOCfailureoccursontheswitchS ,theinverterloses
3
soonasafaultisdetectedinanyoftheswitches.Duetotheir itsfourthvoltagelevel.Inthenegativehalfofthefundamen-
inductance being in series, the parasitic inductance of these tal cycle, this reduces the load current, which reduces the
relays is negligible. From an operational perspective, the current through the FC. Which leads to the loss of inherent
first solution in Fig. 14(a) is the simplest. Convertor recon- capacitorvoltagebalancing.SwitchingR fromtheredundant
2
figuration does not require changing modulation indexes or bridge generates the fourth level. And preserves the output
blowingfuses.Nevertheless,semiconductorsmustbeableto power of the inverter. When OC occurs, switches S , S ,
3 5
withstand the total dc voltage. As a result, the converter is S , R , R , and R are activated with appropriate pulses,
8 2 3 4
considerablymoreexpensive,anditsuseisseverelylimited. resulting in the generation of a three-level output voltage
The second solution in Fig. 14(b) can be useful in some waveform. [64] classifies the SC fault of switches in terms
applications such as controlling an induction motor. It does of the part of the inverter they make SC, including SC of
not require switches that can withstand the total dc voltage, theinputvoltagesourceandSCofthecapacitor.Inthefirst
anditspriceisthelowest.ThethirdsolutioninFig.14(c)like case, because of OC across the input voltage source, a fuse
thepreviousonedoesnotrequireswitchesthatcanwithstand willbeblowntoturntheSCfaultintoanOCfaultwhichis
the total dc voltage, and it is not necessary to reduce the alreadydiscussed.InthesecondcasewhichistheSCofthe
modulation index during the reconfiguration process. This capacitor,theinverteroperatesatequalpowerlevelsasbefore
solutioncanbeagoodoptionforgrid-connectedapplications. thefaultwhilegeneratingathree-levelvoltagewaveformon
Inthetopologyin[63],aredundantlegisaddedtoasingle- theoutput.
phasefive-levelNPC(Fig.15(a)).IftheswitchesS andS The nine-level leg-redundant topology proposed in [65],
a2 b2
failOC,inanNPCwithouttheredundantleg,theconnections consistsoftwothree-levelflyingcapacitorlegsthatarecon-
topointsPandOarenotavailablefortheNPClegs.Hence, nectedbytwocontrolledswitches(Fig.15(c)).bidirectional
theredundantlegRcompensatesfor thisfaultbyusingS switches. It can tolerate OC and SC faults in single and
R1
withS orS toconnecttheinputpointPtolegsAandB multipleswitchesandmaintainstheoutputpowerandvoltage
R2 R4
respectively.ItalsoturnsontheswitchesS withS orS levels in post-fault operation. The switching scheme in this
R6 R2 R4
toconnectpointOtolegsAandB,respectively.Therefore, topology maintains the voltage of the capacitor balanced
thefive-leveloutputvoltagecanbepreserved.Iftheswitches underpre-andpost-faultoperations.
S andS failSC,theNPClegscannotprovideconnections There are also module-level redundancy approaches. For
A1 B1
topointsPandO,andthefusesF andF mustbeblown.The CMCs and MMCs, redundant modules are added in series
1 4
switchescombinationsof(S andS )and(S andS )are with the basic topology as shown in Fig. 16. Normally, the
R1 R2 R1 R4
usedtoconnectthelegAandB,tothepointPrespectively. redundantmodulesareinactive.Wheneveramoduleexperi-
Also,theswitchcombinationsof(S andS )and(S and encesafault,itisisolated,andtheredundantmodulereplaces
R2 R6 R4
S ) can connect the neutral point O to the legs A and B, thefaultymoduletorestorenormaloperation[66].
R6
respectively. Some merits of this inverter are tolerating all Reference [67] investigates the effectiveness of using
typesandlocationsoffaultswithfulloutputratings,reducing redundant cells by the means of reliability assessment and
13438 VOLUME11,2023

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
FIGURE15. Fault-tolerantleg-redundantinvertersproposedin(a)[64],(b)[63],(c)[65].
FIGURE17. Fault-tolerantCHBproposedin[69].
FIGURE16. Fault-tolerantCMCandMMCwithmodule-levelredundancy.
If a switch from the top side and one from the bottom
sidefailtogether,alltheexistingdcsourcesconnecttoform
a set of series-connected dc sources in parallel with the
analyses the cost-effectiveness of designs and redundant redundant module. However, if a second fault happens, all
strategiesofMMC. dcsourcesconnectinserieswhichresultsinasimplethree-
Reference[68]categorizestheredundancystrategiesused level Cascaded Half Bridge (CHB). When one fault occurs,
inmodularmultilevelconvertersinto4approaches:Standard theshapeoftheoutputvoltageremainsunchanged.But,when
Redundancy operation (SR), Redundant operation based on thesecondfaultoccursthenumberoftheoutputvoltagelevel
Additional Submodules (RAS), Redundant operation based reducestothree.
onAdditionalSubmodulesOptimized(RASO),andRedun- The fault-tolerant inverters using the system-level redun-
dantoperationbasedonSpareSubmodules(RSS). dancy are cascaded inverters and parallel inverters. In the
In [69], a fault-tolerant CHB is proposed which uses an cascadedstructureinFig.18(a),twoinvertersareconnected
extra H-bridge module as shown in Fig. 17. The redundant inseries.Althoughinthisapproach,severalfaultsincluding
module just operates after a fault happens. If one of the top single-switch SC, single-switch OC, and phase-leg OC can
switches,inanyoftheH-bridgecellse.g.,switchS1fails,the behandled,thepowerratingisreducedafterthefault[70].
top-siderelaywiththeredcolorinthefigurewillstartfunc- IntheparallelstructureinFig.18(b),Ifoneinverterfails,
tioning. Whenever this relay is triggered, the normal-open theotherinvertercanreplaceitsothatthesystemcanoperate
conductorsoftheinverterwillbeclosed,andtheconductors continuously.However,thereductionofcirculatingcurrents
thatarenormallyclosedwillbeopened,removingthefaulted between converters is an important problem to deal with
componentfromtheinverterandtheredundantmodulejoins when the dual converters are performing simultaneously in
the circuit. when a fault occurs in the bottom switches, the thenormalmode[71].
bottomrelaysinthebluecoloractandtherestoftheaction In the system-level redundant topology in [72] which
isthesameasinthepreviouscase. is shown in Fig. 19, all components including diodes and
VOLUME11,2023 13439

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
| FIGURE18. | System-levelredundantinverters(a)Seriesredundant, |     |     |     |     |     |     |     |     |     |     |
| --------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(b)Parallelredundant.
|     |     |     |     |     | FIGURE20. | Onelegof(a)three-phaseNPCinverter,(b)three-phase |     |     |     |     |     |
| --- | --- | --- | --- | --- | --------- | ------------------------------------------------ | --- | --- | --- | --- | --- |
ANPCinverter.
|     |     |     |     |     | redundant | switches,   | and | altering | the control | strategy, | fault-       |
| --- | --- | --- | --- | --- | --------- | ----------- | --- | -------- | ----------- | --------- | ------------ |
|     |     |     |     |     | tolerance | performance | can | be       | achieved    | which     | will be dis- |
cussedinsectionsIIIandIV.
Inthethree-phaseNPCinverterofwhichtheleg‘‘a’’and
|     |     |     |     |     | its corresponding |           | phase            | is shown | in Fig.    | 20(a),   | three types  |
| --- | --- | --- | --- | --- | ----------------- | --------- | ---------------- | -------- | ---------- | -------- | ------------ |
|     |     |     |     |     | of output         | switching | states           | are      | available, | positive | [P] when     |
|     |     |     |     |     | S a1 and          | S a2      | are on, negative | [N]      | when       | S a3 and | S a4 are on, |
|     |     |     |     |     | and zero          | [O]       | when S           | and S    | are turned | on.      | If S fails   |
FIGURE19. Fault-tolerantseries-resonantinverterproposedin[72]. a2 a3 a2
|     |     |     |     |     | SC, the | negative | state should | be  | avoided | because | it causes |
| --- | --- | --- | --- | --- | ------- | -------- | ------------ | --- | ------- | ------- | --------- |
anSCacrossthebottomdc-bus.IntheSVMtechnique,itis
capacitors are replaceable by turning on the relay R 1 and enough to exclude the vectors with N in their phase ‘‘a’’ as
| turning of     | the R . When                      | one of the switches | S   | to S fails |                                                      |     |     |     |     |     |     |
| -------------- | --------------------------------- | ------------------- | --- | ---------- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                | 2                                 |                     |     | 1 4        | demonstratedinFig.21.Sincethesestatesfallontheoutput |     |     |     |     |     |     |
| SCorOC,switchS | isturnedoff.Thenwiththeassistance |                     |     |            |                                                      |     |     |     |     |     |     |
|                | 4                                 |                     |     |            | perimeterofthehexagon,themaximummodulationindexis    |     |     |     |     |     |     |
ofaseconddiodeD 3 andD 4 ,theremaininghealthyswitches reduced [43], [45]. A similar approach can be implemented
| S andS | arecombinedwiththeredundantswitchesS |     |     | and |                                                  |     |     |     |     |     |     |
| ------ | ------------------------------------ | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
| 1 2    |                                      |     |     | 5   | forSCorOCoftheotherswitchesandalsothediodes.With |     |     |     |     |     |     |
S 6 to reconstruct the inverter with the rated output voltage thisapproach,thefaultiscleared,however,themodification
or power. When a diode fails OC, the faulty diode is auto- of the PWM strategy to avoid unavailable states leads to
| matically | isolated and when | it fails SC, | the fuse | in series |        |           |            |          |     |                  |     |
| --------- | ----------------- | ------------ | -------- | --------- | ------ | --------- | ---------- | -------- | --- | ---------------- | --- |
|           |                   |              |          |           | dc-bus | mid-point | imbalance, | spurious |     | fault detection, | and |
wouldbemelted,thusbypassingit.WhenC 1 experiencesan overratingofdevicevoltagetofulldc-busvoltage[45].
OC,thefaultycapacitorwouldbealsoisolatedautomatically
Inanotherapproach,thepurposeistochangethemodula-
and if SC was experienced, the secondary windings of the tionatthetimeofthefaultinordertomakethesystemsurvive
transformerT willbeshortedmeltingthefuseF .Inthecase theimpact.ActiveNeutralPointClamped(ANPC)converter,
|                                                      | 1   |     | 1   |     |                                                    |     |     |     |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| offailureinjustcapacitorsanddiodes,sinceallswitchesS |     |     |     | to  |                                                    |     |     |     |     |     |     |
|                                                      |     |     |     | 1   | whichisobtainedbyreplacingdiodesinNPCwithswitches, |     |     |     |     |     |     |
S 4 arehealthyinthismode,therearefourpossiblemodulation is widely used in high-power medium-voltage applications
| strategies.Forinstance,inapossibleswitchingset,S |                       |     |      | could |           |             |            |     |         |              |      |
| ------------------------------------------------ | --------------------- | --- | ---- | ----- | --------- | ----------- | ---------- | --- | ------- | ------------ | ---- |
|                                                  |                       |     |      | 3     | including | distributed | generation |     | such as | photovoltaic | sys- |
| beconstantlyon,S                                 | couldbealwaysoff,andS |     | andS |       |           |             |            |     |         |              |      |
4 1 2 could tems,motorcontrolintractionsystems,andindustrialmotor
becomplementary. drives[73].InthisconverterasshowninFig.20(b)[74],[75],
|     |     |     |     |     | ifanOCfaultoccursintheswitchS |     |     |     | a2 ,theswitchesS |     | a3 and |
| --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | ---------------- | --- | ------ |
B. SWITCHINGSTATESREDUNDANCY S canbeturnedontoconnectthephasevoltagetothedc-bus
a6
SwitchingStatesRedundancyincludesavoidingtheunavail- mid-pointwhichminimizestheimpactofthefaultbyreviving
ableswitchingstatesandminimizingtheimpactofthefault thethree-phasesystem.
by a proper switching sequence. The space Vector Modu- In this approach, in case of failure, redundancy in the
lation (SVM) approach is typically used to avoid the states switching states of the inverter, enables the controller to
involvingthefaileddevice.Here,theSVMisrepresentedin chooseanalternateconductionpathtoretainthesameoutput
anα-βframewhichistheconventionaltechnique.TheSVM
voltage[76],[77].
canalsoberepresentedinag-hcoordinatesystemandK-L In the flying capacitor inverter (FC) as shown in Fig. 22,
coordinatesystem. in the normal mode, the voltage level can be provided by
In most cases avoiding the unavailable switching states turningonswitchesS 1 ,S 3 ,andS 4 (thecurrentflowsthrough
may not be enough to obtain the desired performance of the capacitor C and diode D ). If for example the switch
|     |     |     |     |     |     |     | 2   | 2   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
>
a faulted converter. By adding extra components, using S 3 fails open, while i L 0, the same voltage level can be
| 13440 |     |     |     |     |     |     |     |     |     |     | VOLUME11,2023 |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- |

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
|     |     | FIGURE23. | MACtopology,(a)traditionalone,(b)theproposedstructure |     |     |     |
| --- | --- | --------- | ----------------------------------------------------- | --- | --- | --- |
in[78].
FIGURE21. SVMhexagonwithunavailableswitchingstatesasthe TABLE1. Thepossibleoutputvoltagelevelsfortwodifferentswitching
dashedarea.
methodsofthefour-levelMACconverter.
|     |     | switching | states will be | unavailable. | However, | they can be |
| --- | --- | --------- | -------------- | ------------ | -------- | ----------- |
replacedbyotherswitchingstates.
Forexample,toproducelevel1,thereissometimesmore
|     |     | than one | option to replace | the unavailable |     | switching state. |
| --- | --- | -------- | ----------------- | --------------- | --- | ---------------- |
Therefore,therearetwosetmodulationstrategiesinthecase
|     |     | of SC. One      | prioritizes  | the number        | of levels    | and the other |
| --- | --- | --------------- | ------------ | ----------------- | ------------ | ------------- |
|     |     | one prioritizes | the blocking | voltage.          | The produced | voltage       |
|     |     | levels and      | the switches | with overvoltages | are          | demonstrated  |
FIGURE22. Flyingcapacitorinverter.
|     |     | in Table    | 1 for both priorities. | the       | possible       | voltage levels |
| --- | --- | ----------- | ---------------------- | --------- | -------------- | -------------- |
|     |     | are colored | blue. The              | ones that | are achievable | by the new     |
obtained by turning on the switches S and S (the current switching states are colored with a lighter blue. In the case
4 1
flowsthroughthecapacitorC anddiodesD andD ).Onthe oftwofailedswitches,thesituationissimilartotheprevious
|     | 3 2 3 |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- |
other hand, in the healthy condition, turning on the switch casewiththedifferencethatthevoltagelevelswillbeeither
S 2 (the current flows through capacitors C 1 and C 2 and the twoorthreeorevenfourlevels.
diodes D , D , and D ) the voltage level is produced. If S ThetopologyinFig.23(b)iscreatedbyaddingtwoaddi-
| 1 3 | 4   | 3   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
failsshort,whilei <0,thesameoutputvoltageisobtained tional switches to a MAC inverter [78]. This inverter pro-
L
whencurrentflowsthroughdiodesD 1 toD 4 .Therefore,the videsmultipleconductionpathsineachphaseandintra-phase
FC inverter benefits from the switching state redundancy redundancycanbeachievedforcertainswitchingstates.
whichmakesitretainitsoutputvoltagelevelafteranOCor Inaddition,thepost-faultcontrolschemecanbeComplex.
SCfaultoccurs. As a result of its redundancy, this topology is limited to a
Thefour-levelMACconverterdemonstratedinFig.12(a) certain output level or semiconductor device. In the case of
canalwayscontinueoperatingunderasingle-deviceSCand failureinS 11 ,thereisnoalternativepathandlevel1willbe
OC fault maintaining at least three of the four levels using lost. Further, healthy devices may be subjected to increased
redundant switching states. When an SC fault occurs, some blockingvoltagesundercertainfailurescenarios.
| VOLUME11,2023 |     |     |     |     |     | 13441 |
| ------------- | --- | --- | --- | --- | --- | ----- |

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
| FIGURE24. | Voltagephasordiagramforpre-faultandpost-fault. |     |     |     |     |     |     |           |                                               |     |     |     |     |
| --------- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --------- | --------------------------------------------- | --- | --- | --- | --- |
|           |                                                |     |     |     |     |     |     | FIGURE25. | Thevoltagevectorsofamodularmultilevelinverter |     |     |     |     |
(a)Normalcondition,(b)PostfaultafterimplementingNPSapproach.
V. UNBALANCECOMPENSATIONCONTROL
Despitethefactthatthemostimportantfunctionofaninverter
whenafaultoccursistocontinueservicingasclosetonormal
| as possible, | other   | features   | should | also     | be considered   |     | [79].  |     |     |     |     |     |     |
| ------------ | ------- | ---------- | ------ | -------- | --------------- | --- | ------ | --- | --- | --- | --- | --- | --- |
| Imbalance    | control | techniques |        | refer to | the altercation |     | of the |     |     |     |     |     |     |
controlstrategytocorrecttheimbalancescreatedbythefault
andachieveanoptimumoperatingpointconcerningthevolt-
| age, THD,      | or any  | other | objective.     | By  | using   | this algorithm, |     |     |     |     |     |     |     |
| -------------- | ------- | ----- | -------------- | --- | ------- | --------------- | --- | --- | --- | --- | --- | --- | --- |
| fault-tolerant | control | can   | be implemented |     | without | changing        |     |     |     |     |     |     |     |
theinverter’stopology.Asaresult,usingthemcansavehard-
warecostsandsimplifytopologies[80].Threetechniquesof
| control-based | fault | compensation |     | methods | are | discussed | in  |     |     |     |     |     |     |
| ------------- | ----- | ------------ | --- | ------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
thissection.
|     |     |     |     |     |     |     |     | FIGURE26. | Thevoltagevectorsofamodularmultilevelinverter |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------------------------------------------- | --- | --- | --- | --- |
A. PHASESHIFT
(a)Normalcondition,(b)Postfault;theovervoltageissharedamongthree
| In the topologies |     | using | the dc-link | midpoint |     | connection | as  | phases. |     |     |     |     |     |
| ----------------- | --- | ----- | ----------- | -------- | --- | ---------- | --- | ------- | --- | --- | --- | --- | --- |
discussedinsectionII,whenafaultoccurse.g.,onaswitch
intheleg‘‘a’’,thecorrespondingphase‘‘a’’getsconnected (Fig. 25(b)). By solving the following equations, we can
|                 |        |        |                    |         |           |             |        | find the angles | between | phases that | make | the voltage | bal- |
| --------------- | ------ | ------ | ------------------ | ------- | --------- | ----------- | ------ | --------------- | ------- | ----------- | ---- | ----------- | ---- |
| to the midpoint |        | of the | dc-link            | through | turning   | on a TRIAC. |        |                 |         |             |      |             |      |
| The reduced     | system | is     | like a four-switch |         | inverter. | To          | create | anced[83].      |         |             |      |             |      |
balancedline-to-lineoutputvoltages,thephaseangleofthe V =V2+V2−2.V .V .cosα
|     |     |     |     |     |     |     |     |     | ab  | a b | a   | b   | (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
healthyphasevoltages(bandc)shouldbeadjustedbyshift-
|                                                |     |     |     |     |     |     |     |     | =V2+V2−2.V |     | .V  | .cosβ |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | ----- | --- |
| ingby30degreeswhichisdemonstratedinFig.24[81]. |     |     |     |     |     |     |     |     | V bc       |     | b   | c     | (2) |
b c
|     |     |     |     |     |     |     |     |     | =V2+V2−2.V |     | .V  | .cosγ |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | ----- | --- |
Inthecasethattheneutralpointofthethree-phasemotor V ca c a (3)
c a
| is connected | to  | the dc-bus | midpoint, |     | the phase | angle | of  |     |         |          |     |     |     |
| ------------ | --- | ---------- | --------- | --- | --------- | ----- | --- | --- | ------- | -------- | --- | --- | --- |
|              |     |            |           |     |           |       |     |     | V ab =V | bc =V ca |     |     | (4) |
the healthy phase currents (b and c) should be shifted by α+β+γ ◦
|     |     |     |     |     |     |     |     |     | =360 |     |     |     | (5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
30degrees[82].
|                 |     |           |     |         |           |           |     | As shown               | in Fig. 25(b), | the output | voltage | is higher | than |
| --------------- | --- | --------- | --- | ------- | --------- | --------- | --- | ---------------------- | -------------- | ---------- | ------- | --------- | ---- |
| B. NEUTRALSHIFT |     |           |     |         |           |           |     | theconventionalmethod. |                |            |         |           |      |
| When bypassing  |     | a module, | the | voltage | and power | available |     |                        |                |            |         |           |      |
from the drive are reduced, but the available current is not C. EXTENDEDNEUTRALSHIFT
affected. As it was mentioned in section II, when a fault This method is applied in cases in which the converter’s
|           |          |     |        |         |     |        |       | neutral point | obtained | through the | traditional | neutral-shift |     |
| --------- | -------- | --- | ------ | ------- | --- | ------ | ----- | ------------- | -------- | ----------- | ----------- | ------------- | --- |
| occurs in | a module | in  | an MMC | or CMC, | one | option | after |               |          |             |             |               |     |
isolationofthefaultymoduleisisolatingthecorresponding approach is located outside the triangle of the output line-
modules in the other two phases to keep the output voltage to-linevoltages[84].Usingthisapproach,theanglebetween
balanced.However,theoutputvoltageisreduced. the two voltages with the lowest amplitude is calculated at
ByusingtheNPSmethod,thereisnoneedtobypassthe 180degrees,andtheamplitudeandangleoftheotherphase
|               |         |     |         |          |            |     |        | are calculated | to maximize | the output | voltage. | This | method |
| ------------- | ------- | --- | ------- | -------- | ---------- | --- | ------ | -------------- | ----------- | ---------- | -------- | ---- | ------ |
| corresponding | healthy |     | modules | and have | a balanced |     | output |                |             |            |          |      |        |
at the same time. As shown in Fig. 25(a), the line-to-line canincreasetheoutputvoltageby15%.
voltagesinnormaloperationare8.67p.u.
When modules W 4 , W 5 , and V 5 experience a fault, the D. VOLTAGEEXTENSION
correspondent healthy modules, which are V , U , and U This method is similar to the NPS technique with the dif-
|     |     |     |     |     |     | 4 4 | 5   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
are bypassed. The new line-to-line voltages are 5.19 p.u. ferencethat theoutput voltagecan besustainedat thesame
| 13442 |     |     |     |     |     |     |     |     |     |     |     | VOLUME11,2023 |     |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- |

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
levelasthatinthepre-faultcondition[85],[86].Animportant first depends on the type of inverter and the application in
drawbackofthepreviousreconfigurationstrategyisitseffect which the inverter needs to be fault tolerant. Since usually
onthecommonmodevoltage,whichcanleadtounbearable extracomponentsareaddedinfaulttolerancestrategies,the
stress on the machine bearings [84]. Many papers includ- cost is an important factor in considering the desired fault
ing[87],[88],[89],[90],[91],[92],[93],[94]haveusedthis compensationsolutionalongwithLossesandcomplexity.
methodintheirfault-tolerantoperationscheme. Althoughvariousfault-toleranttopologieshavebeenpro-
To increase the converter’s maximum output range, the posed in the last two decades, due to the increase in the
averageofthemaximumandminimumreferencephasevolt- demand for reliable systems, there are lots of potentials to
ages is injected into the common-mode voltages. Fig. 26(a) proposenovelfaultcompensationideas,especiallyimbalance
shows the phasor diagram of the normal operation. When a controlstrategies.
faultoccurs,inordertomaintaintheoutputvoltagelevel,the As users may have different main considerations, the
inputdc-busvoltageofthefaultyphaseisincreasedinorder suitable redundant designs are application dependent. For
tokeepthetotalvoltageunchanged.AsshowninFig.26(b), a noncritical application, the economy is the main consid-
the three voltages are balanced, and their value is the same eration. Although using redundant cells increase the cost
asthenormaloperation.However,themodulesinthephase of the system, a high level of reliability and survivability
whichhadthefaultymodules,experienceovervoltage.There- of the drive system is essential in some critical industrial
fore, to equally share the increased voltage burden among processesinvolvinghighstandstillcostsandsafetyconcerns.
all healthy modules of three phases, optimal angles of the In addition, lifetime prediction and cost assessment enable
phasevoltagesarecalculatedbyequations(1)to(5)toobtain ustoidentifyredundantdesignsthataremostcost-effective.
Fig.26(b). Despitetheabundanceofredundantdesignsandfault-tolerant
Theoutputharmonicdistortion,however,maybesignifi- algorithms, their reliability improvements remain largely
cant since this method requires the converter to work in the unquantified.Redundancycostsareassessedtodeterminethe
overmodulation region by injecting excess common-mode costoffault-tolerantconvertersinordertoprovidemanufac-
voltage [50]. Even though common-mode voltages do not turerswithanaffordableoption.
appear in output line-line voltages, they can lead to unbear-
able voltage stress on motor bearings and shafts. With this
REFERENCES
method, the fundamental amplitude of the common mode
[1] Y.Yang,H.Wang,A.Sangwongwanich,andF.Blaabjerg,‘‘Designfor
voltage is reduced, resulting in a more bearable operating reliabilityofpowerelectronicsystems,’’inPowerElectronicsHandbook.
conditionfortheload[84]. Amsterdam,TheNetherlands:Elsevier,2018,pp.1423–1440.
Not only are there fault conditions without an equation [2] J. Falck, C. Felgemacher, A. Rojko, M. Liserre, and P. Zacharias,
‘‘Reliability of power electronic systems: An industry perspective,’’
solutionoruniquesolution,buttheymayalsonotcausethe
IEEE Ind. Electron. Mag., vol. 12, no. 2, pp.24–35, Jun. 2018, doi:
maximumavailablevoltage. 10.1109/mie.2018.2825481.
[3] S. Peyghami, F. Blaabjerg, and S. Kaboli, Resilient Power Electronic
Systems.Hoboken,NJ,USA:Wiley,2022.
VI. CONCLUSION
[4] Q.Sun,X.Yu,H.Li,andJ.Fan,‘‘Adaptivefeatureextractionandfault
In fault-tolerant topologies, due to the increased number of
diagnosisforthree-phaseinverterbasedonhybrid-CNNmodelsundervari-
switches, fault-tolerant converters are more likely to have ableoperatingconditions,’’ComplexIntell.Syst.,vol.8,no.1,pp.29–42,
switchbreakdownsthanstandardconverters.However,when Feb.2022,doi:10.1007/s40747-021-00337-6.
[5] S. Bouguerra, M. R. Yaiche, O. Gassab, A. Sangwongwanich, and
thesesolutionsareused,thereisasignificantdecreaseinthe
F. Blaabjerg, ‘‘The impact of PV panel positioning and degradation
probabilityofacompleteconverterfailure,somethingthatis on the PV inverter lifetime and reliability,’’ IEEE J. Emerg. Sel. Top-
imperativeformanyindustriesthatusepowerelectroniccon- ics Power Electron., vol. 9, no. 3, pp.3114–3126, Jun. 2021, doi:
10.1109/JESTPE.2020.3006267.
verters.Thefirststepoffaultmanagementisfaultdiagnosis
[6] S.Peyghami,P.Palensky,andF.Blaabjerg,‘‘Anoverviewonthereliability
and after that, fault isolation is considered the primary step ofmodernpowerelectronicbasedpowersystems,’’IEEEOpenJ.Power
to minimize the aftermath of a fault by isolating the fault Electron.,vol.1,pp.34–50,2020,doi:10.1109/OJPEL.2020.2973926.
using extra hardware including fuses, TRIACs, etc. After [7] M.Fahad,M.Tariq,A.Sarwar,M.Modabbir,M.A.Zaid,K.Satpathi,
M.R.Hussan,M.Tayyab,B.Alamri,andA.Alahmadi,‘‘Asymmetric
faultisolation,theeffectsofthefaultmustbecompensated.
multilevelinvertertopologyanditsfaultmanagementstrategyforhigh-
The switching state redundancy techniques may use a few reliabilityapplications,’’Energies,vol.14,no.14,p.4302,Jul.2021,doi:
extra components to make the possibility of using alternate 10.3390/en14144302.
[8] L.F.CostaandM.Liserre,‘‘FailureanalysisoftheDC–DCconverter:
conductionpathsforpost-fault,however,theseextracompo-
Acomprehensivesurveyoffaultsandsolutionsforimprovingreliability,’’
nent does not necessarily put them in the hardware redun- IEEE Power Electron. Mag., vol. 5, no. 4, pp.42–51, Dec. 2018, doi:
dancycategorywhichusuallyusestheextraswitchesorlegs 10.1109/MPEL.2018.2874345.
or modules to replace the faulty ones directly. When a fault [9] Z. Li, J. Xia, Y. Guo, and X. Zhang, ‘‘Fault-tolerant predictive torque
controldesignforinductionmotordrivesbasedondiscretespacevector
happensinthe inverter,orevenafter usingafaulthardware
modulation,’’IEEEJ.Emerg.Sel.TopicsPowerElectron.,vol.9,no.5,
redundancy or switching states redundancy methods, there pp.5441–5451,Oct.2021,doi:10.1109/JESTPE.2021.3064979.
may be imbalances such as voltage unbalance that can be [10] M.Fahad,M.Alsultan,S.Ahmad,A.Sarwar,M.Tariq,andI.A.Khan,
‘‘Reliabilityanalysisandfault-tolerantoperationinamultilevelinverter
taken care of using enhancing the control strategy of the
forindustrialapplication,’’Electronics,vol.11,no.1,p.98,Dec.2021,
converter. Choosing the best fault compensation technique doi:10.3390/electronics11010098.
VOLUME11,2023 13443

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
[11] F.Blaabjerg,H.Wang,I.Vernica,B.Liu,andP.Davari,‘‘Reliabilityof [30] G.K.KumarandD.Elangovan,‘‘Reviewonfault-diagnosisandfault-
powerelectronicsystemsforEV/HEVapplications,’’Proc.IEEE,vol.109, toleranceforDC–DCconverters,’’IETPowerElectron.,vol.13,no.1,
no.6,pp.1060–1076,Jun.2021,doi:10.1109/JPROC.2020.3031041. pp.1–13,Jan.2020,doi:10.1049/iet-pel.2019.0672.
[12] M. Omana, A. Fiore, M. Mongitore, and C. Metra, ‘‘Fault-tolerant [31] Z.Li,P.Wheeler,A.Watson,A.Costabeber,B.Wang,Y.Ren,Z.Bai,
inverters for reliable photovoltaic systems,’’ IEEE Trans. Very Large andH.Ma,‘‘AfastdiagnosismethodforbothIGBTfaultsandcurrent
Scale Integr. (VLSI) Syst., vol. 27, no. 1, pp.20–28, Jan. 2019, doi: sensorfaultsingrid-tiedthree-phaseinverterswithtwocurrentsensors,’’
10.1109/TVLSI.2018.2874709. IEEETrans.PowerElectron.,vol.35,no.5,pp.5267–5278,May2020,
[13] D.Kumar,R.K.Nema,andS.Gupta,‘‘Developmentofanovelfault- doi:10.1109/TPEL.2019.2946692.
tolerant reduced device count T-type multilevel inverter topology,’’ Int. [32] J.He,Q.Yang,andZ.Wang,‘‘On-linefaultdiagnosisandfault-tolerant
J.Electr.PowerEnergySyst.,vol.132,Nov.2021,Art.no.107185,doi: operationofmodularmultilevelconverters—Acomprehensivereview,’’
10.1016/j.ijepes.2021.107185. CESTrans.Electr.Mach.Syst.,vol.4,no.4,pp.360–372,Dec.2020,doi:
10.30941/cestems.2020.00043.
[14] D.KrishnachaitanyaandA.Chitra,‘‘Quantitativeanalysisofasymmetric
|     |     |     |     |     |     |     | [33] B. Gou, | Y. Xu, | Y. Xia, | Q. Deng, | and X. Ge, | ‘‘An online | data-driven |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | ------- | -------- | ---------- | ----------- | ----------- |
multilevelinverterswithreduceddevicecountfromreliabilityandcost
functionperspective—Areview,’’IEEETrans.PowerElectron.,vol.36, method for simultaneous diagnosis of IGBT and current sensor fault
no.10,pp.11068–11086,Oct.2021,doi:10.1109/TPEL.2021.3071375. of three-phase PWM inverter in induction motor drives,’’ IEEE Trans.
[15] D. Vinnikov, A. Chub, D. Zinchenko, V. Sidorov, M. Malinowski, and Power Electron., vol. 35, no. 12, pp.13281–13294, Dec. 2020, doi:
10.1109/TPEL.2020.2994351.
S.Bayhan,‘‘Topology-morphingphotovoltaicmicroconverterwithwide
|     |     |     |     |     |     |     | [34] B.LuandS.Sharma,‘‘AliteraturereviewofIGBTfaultdiagnosticand |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
MPPTvoltagewindowandpost-faultoperationcapability,’’IEEEAccess,
protectionmethodsforpowerinverters,’’IEEETrans.Ind.Appl.,vol.45,
vol.8,pp.153941–153955,2020,doi:10.1109/ACCESS.2020.3017805.
no.5,pp.1770–1777,Nov.2009,doi:10.1109/TIA.2009.2027535.
| [16] S. Rahimpour, | H.  | Tarzamni, | N. V. Kurdkandi, | O.  | Husev, D. | Vinnikov, |              |            |     |             |            |          |               |
| ------------------ | --- | --------- | ---------------- | --- | --------- | --------- | ------------ | ---------- | --- | ----------- | ---------- | -------- | ------------- |
|                    |     |           |                  |     |           |           | [35] Z. Gao, | C. Cecati, | and | S. X. Ding, | ‘‘A survey | of fault | diagnosis and |
andF.Tahami,‘‘Anoverviewoflifetimemanagementofpowerelectronic
converters,’’IEEEAccess,vol.10,pp.109688–109711,2022. fault-toleranttechniques—PartI:Faultdiagnosiswithmodel-basedand
|     |     |     |     |     |     |     | signal-based | approaches,’’ |     | IEEE Trans. | Ind. | Electron., | vol. 62, no. 6, |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------- | --- | ----------- | ---- | ---------- | --------------- |
[17] S.Ye,J.Jiang,J.Li,Y.Liu,Z.Zhou,andC.Liu,‘‘Faultdiagnosisand
pp.3757–3767,Jun.2015,doi:10.1109/TIE.2015.2417501.
tolerancecontroloffive-levelnestedNPPconverterusingwaveletpacket
|     |     |     |     |     |     |     | [36] Z.Gao,C.Cecati,andS.Ding,‘‘Asurveyoffaultdiagnosisandfault- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
andLSTM,’’IEEETrans.PowerElectron.,vol.35,no.2,pp.1907–1921,
|     |     |     |     |     |     |     | tolerant | techniques | Part | II: Fault diagnosis | with | knowledge-based | and |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ---- | ------------------- | ---- | --------------- | --- |
Feb.2020,doi:10.1109/TPEL.2019.2921677.
|     |     |     |     |     |     |     | hybrid/active |     | approaches,’’ | IEEE Trans. | Ind. | Electron., | vol. 62, no. 6, |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------- | ----------- | ---- | ---------- | --------------- |
[18] J.M.S.Callegari,A.F.Cupertino,V.D.N.Ferreira,andH.A.Pereira,
pp.3768–3774,Jun.2015,doi:10.1109/TIE.2015.2419013.
‘‘MinimumDC-linkvoltagecontrolforefficiencyandreliabilityimprove- [37] R. L. D. A. Ribeiro, C. B. Jacobina, E. R. C. D. Silva, and
| ment in | PV inverters,’’ | IEEE | Trans. Power | Electron., | vol. | 36, no. 5, |       |          |                  |             |     |              |          |
| ------- | --------------- | ---- | ------------ | ---------- | ---- | ---------- | ----- | -------- | ---------------- | ----------- | --- | ------------ | -------- |
|         |                 |      |              |            |      |            | A. M. | N. Lima, | ‘‘Fault-tolerant | voltage-fed |     | PWM inverter | AC motor |
pp.5512–5520,May2021,doi:10.1109/TPEL.2020.3032040.
drivesystems,’’IEEETrans.Ind.Electron.,vol.51,no.2,pp.439–446,
[19] A.Haque,FaultAnalysisanditsImpactonGrid-ConnectedPhotovoltaic
Apr.2004,doi:10.1109/TIE.2004.825284.
SystemsPerformance.Hoboken,NJ,USA:Wiley,2022.
|                    |     |            |              |     |           |              | [38] J.-R.FuandT.A.Lipo,‘‘Astrategytoisolateaswitchingdevicefaultin |     |     |     |     |     |     |
| ------------------ | --- | ---------- | ------------ | --- | --------- | ------------ | ------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| [20] M. Kamalirad, | H.  | Iman-Eini, | B. Farhangi, | and | S. Bacha, | ‘‘A reliable |                                                                     |     |     |     |     |     |     |
acurrentregulatedmotordrive,’’Electr.Mach.PowerSyst.,vol.24,no.8,
three-phasetransformerlessgrid-connectedPVinverterwithinductiveDC pp.911–920,Dec.1996.
link,’’IEEEJ.Photovolt.,vol.8,no.5,pp.1305–1312,Sep.2018,doi: [39] Y.SongandB.Wang,‘‘Analysisandexperimentalverificationofafault-
10.1109/JPHOTOV.2018.2846702.
tolerantHEVpowertrain,’’IEEETrans.PowerElectron.,vol.28,no.12,
[21] K.Li,S.Cheng,T.Yu,X.Wu,C.Xiang,andA.Bilal,‘‘Anon-linemultiple
pp.5854–5864,Dec.2013,doi:10.1109/TPEL.2013.2245513.
open-circuitfaultdiagnostictechniqueforrailwayvehicleair-conditioning
|              |             |      |           |          |                      |     | [40] S.Bolognani,M.Zordan,andM.Zigliotto,‘‘Experimentalfault-tolerant |           |          |             |      |            |                 |
| ------------ | ----------- | ---- | --------- | -------- | -------------------- | --- | --------------------------------------------------------------------- | --------- | -------- | ----------- | ---- | ---------- | --------------- |
| inverters,’’ | IEEE Trans. | Veh. | Technol., | vol. 69, | no. 7, pp.7026–7039, |     |                                                                       |           |          |             |      |            |                 |
|              |             |      |           |          |                      |     | control                                                               | of a PMSM | drive,’’ | IEEE Trans. | Ind. | Electron., | vol. 47, no. 5, |
Jul.2020,doi:10.1109/TVT.2020.2987935. pp.1134–1141,Oct.2000.
[22] D. Vinnikov, A. Chub, O. Korkh, and M. Malinowski, ‘‘Fault-tolerant [41] S.Ceballos,J.Pou,E.Robles,J.Zaragoza,andJ.L.Martin,‘‘Three-leg
bidirectional series resonant DC–DC converter with minimum number fault-tolerantneutral-point-clampedconverter,’’inProc.IEEEInt.Symp.
ofcomponents,’’inProc.IEEEEnergyConvers.Congr.Expo.(ECCE),
Ind.Electron.,Jun.2007,pp.3180–3185.
Sep.2019,pp.1359–1363,doi:10.1109/ECCE.2019.8912292.
|     |     |     |     |     |     |     | [42] S. Ceballos, | J.  | Pou, E. | Robles, J. | Zaragoza, | and J. L. | Martín, ‘‘Per- |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------- | ---------- | --------- | --------- | -------------- |
[23] J.Chen,C.Zhang,A.Chen,andX.Xing,‘‘Fault-tolerantcontrolstrate-
formanceevaluationoffault-tolerantneutral-point-clampedconverters,’’
gies for T-type three-level inverters considering neutral-point voltage IEEETrans.Ind.Electron.,vol.57,no.8,pp.2709–2718,Aug.2010,doi:
oscillations,’’IEEETrans.Ind.Electron.,vol.66,no.4,pp.2837–2846, 10.1109/TIE.2009.2026710.
Apr.2019,doi:10.1109/TIE.2018.2842731. [43] S. Li and L. Xu, ‘‘Strategies of fault tolerant operation for three-level
[24] N.K.Dewangan,T.Prakash,J.K.Tandekar,andK.K.Gupta,‘‘Open- PWMinverters,’’IEEETrans.PowerElectron.,vol.21,no.4,pp.933–940,
| circuit fault-tolerance |     | in multilevel | inverters | with | reduced | component |     |     |     |     |     |     |     |
| ----------------------- | --- | ------------- | --------- | ---- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
Jul.2006,doi:10.1109/TPEL.2006.876867.
| count,’’ Electr. | Eng., | vol. | 102, no. 1, | pp.409–419, | Mar. | 2020, doi: |                  |     |            |                |     |           |              |
| ---------------- | ----- | ---- | ----------- | ----------- | ---- | ---------- | ---------------- | --- | ---------- | -------------- | --- | --------- | ------------ |
|                  |       |      |             |             |      |            | [44] S. Farnesi, | P.  | Fazio, and | M. Marchesoni, | ‘‘A | new fault | tolerant NPC |
10.1007/s00202-019-00884-9. converter system for high power induction motor drives,’’ in Proc. 8th
[25] X. Guo, S. Sui, B. Wang, and W. Zhang, ‘‘A current-based approach IEEE Symp. Diag. Electr. Mach., Power Electron. Drives, Sep. 2011,
| for short-circuitfault |     | diagnosisin | closed-loopcurrent |     | sourceinverter,’’ |     | pp.337–343. |     |     |     |     |     |     |
| ---------------------- | --- | ----------- | ------------------ | --- | ----------------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
IEEETrans.Ind.Electron.,vol.67,no.9,pp.7941–7950,Sep.2020,doi: [45] J.-J.Park,T.-J.Kim,andD.-S.Hyun,‘‘Studyofneutralpointpotential
10.1109/TIE.2019.2941143.
variationforthree-levelNPCinverterunderfaultcondition,’’inProc.34th
[26] A. Blinov, R. Kosenko, A. Chub, and V. Ivakhno, ‘‘Analysis of fault- Annu.Conf.IEEEInd.Electron.,Nov.2008,pp.983–988.
tolerantoperationcapabilitiesofanisolatedbidirectionalcurrent-source [46] Z.Ni,A.H.Abuelnaga,andM.Narimani,‘‘Anewfault-toleranttech-
DC–DCconverter,’’Energies,vol.12,no.16,p.3203,Aug.2019,doi: niquebasedonnonsymmetricalselectiveharmoniceliminationforcas-
10.3390/en12163203. cadedH-bridgemotordrives,’’IEEETrans.Ind.Electron.,vol.68,no.6,
[27] Z.Huang,Z.Wang,andL.Liu,‘‘Apracticalfaultdiagnosisalgorithm pp.4610–4622,Jun.2020.
basedonaperiodiccorrected-secondlow-frequencyprocessingformicro- [47] J. A. V. M. Farias, A. F. Cupertino, H. A. Pereira, S. I. Seleme, and
gridinverter,’’IEEETrans.Ind.Informat.,vol.15,no.7,pp.3889–3898,
R.Teodorescu,‘‘OnconverterfaulttoleranceinMMC-HVDCsystems:A
Jul.2019,doi:10.1109/TII.2018.2883768. comprehensivesurvey,’’IEEEJ.Emerg.Sel.TopicsPowerElectron.,vol.9,
[28] B.Wang,Z.Li,Z.Bai,P.T.Krein,andH.Ma,‘‘Aredundantunittoform no.6,pp.7459–7470,Dec.2021,doi:10.1109/JESTPE.2020.3032393.
T-Typethree-levelinverterstolerantofIGBTopen-circuitfaultsinmultiple [48] G.Chen,A.Bahrami,andM.Narimani,‘‘Anewseven-leveltopologyfor
legs,’’IEEETrans.PowerElectron.,vol.35,no.1,pp.924–939,Jan.2020, high-powermedium-voltageapplication,’’IEEETrans.Ind.Electron.,vol.
doi:10.1109/TPEL.2019.2912177. 68,no.1,pp.37–46,Jan.2021,doi:10.1109/TIE.2019.2962456.
[29] A.Moradmand,M.Dorostian,A.Ramezani,A.Sajadi,andB.Shafai, [49] N. Bisht and A. Das, ‘‘A multiple fault-tolerant topology of cascaded
‘‘Fault-tolerant control of inverter for the integration of solar PV H-bridgeconverterformotordrivesusingexistingprechargewindings,’’
underabnormalconditions,’’J.Eng.,vol.2020,no.11,pp.1112–1122, IEEEJ.Emerg.Sel.TopicsPowerElectron.,vol.9,no.2,pp.2079–2087,
Nov.2020,doi:10.1049/joe.2019.1280. Apr.2021,doi:10.1109/JESTPE.2020.3002830.
| 13444 |     |     |     |     |     |     |     |     |     |     |     |     | VOLUME11,2023 |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- |

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
[50] Z.Ni,A.H.Abuelnaga,andM.Narimani,‘‘Anewfault-toleranttech- [68] J. V. M. Farias, A. F. Cupertino, H. A. Pereira, S. I. S. Junior, and
niquebasedonnonsymmetricalselectiveharmoniceliminationforcas- R.Teodorescu,‘‘Ontheredundancystrategiesofmodularmultilevelcon-
cadedH-bridgemotordrives,’’IEEETrans.Ind.Electron.,vol.68,no.6, verters,’’IEEETrans.PowerDel.,vol.33,no.2,pp.851–860,Jun.2018,
pp.4610–4622,Jun.2021,doi:10.1109/TIE.2020.2989705. doi:10.1109/TPWRD.2017.2713394.
[51] M. Hassanifar, D. Nazarpour, S. Golshannavaz, and Y. Neyshabouri, [69] H. K. Jahan, F. Panahandeh, M. Abapour, and S. Tohidi, ‘‘Recon-
‘‘A modular cascaded multilevel converter with high configurability: figurable multilevel inverter with fault-tolerant ability,’’ IEEE Trans.
Design, analysis, and optimization study,’’ IEEE J. Emerg. Sel. Top- Power Electron., vol. 33, no. 9, pp.7880–7893, Sep. 2018, doi:
ics Power Electron., vol. 11, no. 1, pp.850–861, Feb. 2023, doi: 10.1109/TPEL.2017.2773611.
10.1109/JESTPE.2021.3114501. [70] B.A.Welchko,T.A.Lipo,T.M.Jahns,andS.E.Schulz,‘‘Faulttolerant
[52] M. M. Haji-Esmaeili, M. Naseri, H. Khoun-Jahan, and M. Abapour, three-phaseACmotordrivetopologies:Acomparisonoffeatures,cost,and
‘‘Fault-tolerant structure for cascaded H-bridge multilevel inverter and limitations,’’IEEETrans.PowerElectron.,vol.19,no.4,pp.1108–1116,
reliability evaluation,’’ IET Power Electron., vol. 10, no. 1, pp.59–70, Jul.2004.
Jan.2017,doi:10.1049/iet-pel.2015.1025. [71] L.Asiminoaei,E.Aeloiza,P.N.Enjeti,andF.Blaabjerg,‘‘Shuntactive-
[53] S.Ouni,A.U.Schmeisser,M.Zolghadri,H.Oraee,J.Rodriguez,and power-filter topology based on parallel interleaved inverters,’’ IEEE
P.Lezana,‘‘Adecisionalgorithmtoselectapropercontrolmethodfora Trans. Ind. Electron., vol. 55, no. 3, pp.1175–1189, Jan. 2008, doi:
cascadedmultilevelinverterunderfaultycondition,’’inProc.41stAnnu. 10.1109/TIE.2007.907671.
Conf.IEEEInd.Electron.Soc.(IECON),Nov.2015,pp.004830–004835. [72] J.Huang,G.Chen,andH.Shi,‘‘Acost-reliabilitytrade-offfault-tolerant
[54] J. Aguayo-Alquicira, I. Vásquez-Libreros, S. E. De Léon-Aldaco, series-resonant converter combining redundancy and reconstruction,’’
M. Ponce-Silva, R. E. Lozoya-Ponce, E. Flores-Rodríguez, J. García- IEEETrans.PowerElectron.,vol.36,no.10,pp.11543–11554,Oct.2021,
Morales,Y.Reyes-Severiano,L.M.Carrillo-Santos,M.Marín-Reyes,and doi:10.1109/TPEL.2021.3072383.
E.M.Amores-Campos,‘‘Reconfigurationstrategyforfaulttolerancein [73] T. J. Nistane, L. K. Sahu, M. Jalhotra, and S. P. Gautam, ‘‘Single and
acascadedmultilevelinverterusingaZ-sourceconverter,’’Electronics, multipleswitchfault-tolerancecapabilitiesinahybridfive-levelinverter
vol.10,no.5,p.574,Mar.2021,doi:10.3390/electronics10050574. topology,’’IETPowerElectron.,vol.13,no.6,pp.1257–1266,May2020,
[55] A. L. Julian and G. Oriti, ‘‘A comparison of redundant inverter doi:10.1049/iet-pel.2019.0716.
topologies to improve voltage source inverter reliability,’’ IEEE Trans. [74] T.Bruckner,S.Bernet,andH.Guldner,‘‘TheactiveNPCconverterand
Ind. Appl., vol. 43, no. 5, pp.1371–1378, Sep./Oct. 2007, doi: its loss-balancing control,’’ IEEE Trans. Ind. Electron., vol. 52, no. 3,
10.1109/TIA.2007.904436. pp.855–868,Jun.2005,doi:10.1109/TIE.2005.847586.
[56] X.Kou,K.A.Corzine,andY.L.Familiant,‘‘Auniquefault-tolerantdesign [75] J.Li,A.Q.Huang,Z.Liang,andS.Bhattacharya,‘‘Analysisanddesign
for flying capacitor multilevel inverter,’’ IEEE Trans. Power Electron., of active NPC (ANPC) inverters for fault-tolerant operation of high-
vol.19,no.4,pp.979–987,Jul.2004,doi:10.1109/TPEL.2004.830037. power electrical drives,’’ IEEE Trans. Power Electron., vol. 27, no. 2,
[57] A.Chen,C.Zhang,X.He,andN.Cui,‘‘Fault-tolerantdesignforflying pp.519–533,Feb.2012,doi:10.1109/TPEL.2011.2143430.
capacitor multilevel inverters,’’ in Proc. IEEE 6th Int. Power Electron. [76] D. Kumar, R. K. Nema, and S. Gupta, ‘‘Investigation of fault-
MotionControlConf.,May2009,pp.1460–1464. tolerant capabilities of some recent multilevel inverter topologies,’’
[58] A.Cordeiro,J.Palma,J.Maia,andM.Resende,‘‘Combiningmechani- Int. J. Electron., vol. 108, no. 11, pp.1957–1976, Nov. 2021, doi:
calcommutatorsandsemiconductorsinfastchangingredundantinverter 10.1080/00207217.2020.1870752.
topologies,’’ in Proc. Int. Conf. Comput. Tool (IEEE EUROCON), [77] S. Tang, X. Yin, D. Wang, C. Zhang, Z. Shuai, X. Yang,
Apr.2011,pp.1–4. Z. J. Shen, and J. Wang, ‘‘Detection and identification of power
[59] A. Madhukar Rao and K. Sivakumar, ‘‘A fault-tolerant single-phase switchfailuresforfault-tolerantoperationofflyingcapacitorbuck-boost
five-level inverter for grid-independent PV systems,’’ IEEE Trans. converters,’’Microelectron.Rel.,vols.88–90,pp.1236–1241,Sep.2018,
Ind. Electron., vol. 62, no. 12, pp.7569–7577, Dec. 2015, doi: doi:10.1016/j.microrel.2018.06.102.
10.1109/TIE.2015.2455523. [78] J.Nicolas-Apruzzese,S.Busquets-Monge,J.Bordonau,S.Alepuz,and
[60] P.Weber,P.Poure,D.Theilliol,andS.Saadate,‘‘Designofhardwarefault A.Calle-Prado,‘‘Analysisofthefault-tolerancecapacityofthemultilevel
tolerantcontrolarchitectureforwindenergyconversionsystemwithDFIG active-clampedconverter,’’IEEETrans.Ind.Electron.,vol.60,no.11,
based on reliability analysis,’’ in Proc. IEEE Int. Symp. Ind. Electron., pp.4773–4783,Nov.2013,doi:10.1109/TIE.2012.2222856.
Jun.2008,pp.2323–2328. [79] S. Ouni et al., ‘‘Improvement of post-fault performance of a cascaded
[61] V. V. S. P. Kumar and B. G. Fernandes, ‘‘A fault-tolerant single-phase H-bridgemultilevelinverter,’’IEEETrans.Ind.Electron.,vol.64,no.4,
grid-connectedinvertertopologywithenhancedreliabilityforsolarPV pp.2779–2788,Nov.2016.
applications,’’IEEEJ.Emerg.Sel.Top.PowerElectron.,vol.5,no.3, [80] G. Li, C. Liu, and Y. Wang, ‘‘Voltage space vector equivalent substi-
pp.1254–1262,Sep.2017,doi:10.1109/JESTPE.2017.2687126. tution fault-tolerance control for cascaded H-bridge multilevel inverter
[62] S.Ceballos,J.Pou,J.Zaragoza,E.Robles,J.L.Villate,andJ.L.Martin, withcurrent-tracking,’’Electronics,vol.9,no.1,p.93,Jan.2020,doi:
‘‘Fault-tolerantneutral-point-clampedconvertersolutionsbasedoninclud- 10.3390/electronics9010093.
ing a fourth resonant leg,’’ IEEE Trans. Ind. Electron., vol. 58, no. 6, [81] S. Xu, J. Zhang, and J. Hang, ‘‘Investigation of a fault-tolerant three-
pp.2293–2303,Jun.2011,doi:10.1109/TIE.2010.2069075. level T-type inverter system,’’ IEEE Trans. Ind. Appl., vol. 53, no. 5,
[63] M. Aly, E. Ahmed, and M. Shoyama, ‘‘A new single-phase five-level pp.4613–4623,Sep./Oct.2017,doi:10.1109/TIA.2017.2697844.
invertertopologyforsingleandmultipleswitchesfaulttolerance,’’IEEE [82] S.R.MadetiandS.N.Singh,‘‘Acomprehensivestudyondifferenttypesof
Trans.PowerElectron.,vol.33,no.11,pp.9198–9208,Nov.2018,doi: faultsanddetectiontechniquesforsolarphotovoltaicsystem,’’Sol.Energy,
10.1109/TPEL.2018.2792146. vol.158,pp.161–185,Dec.2017,doi:10.1016/j.solener.2017.08.069.
[64] M.Jalhotra,L.K.Sahu,S.Gupta,andS.P.Gautam,‘‘Highlyresilient [83] J. Rodriguez, P. W. Hammond, J. Pontt, R. Musalem, P. Lezana, and
fault-toleranttopologyofsingle-phasemultilevelinverter,’’IEEEJ.Emerg. M.J.Escobar,‘‘Operationofamedium-voltagedriveunderfaultycon-
Sel.TopicsPowerElectron.,vol.9,no.2,pp.1915–1922,Apr.2021,doi: ditions,’’ IEEE Trans. Ind. Electron., vol. 52, no. 4, pp.1080–1085,
10.1109/JESTPE.2019.2936271. Aug.2005.
[65] A. Chappa, S. Gupta, L. K. Sahu, and K. K. Gupta, ‘‘A fault-tolerant [84] P.LezanaandG.Ortiz,‘‘Extendedoperationofcascademulticellcon-
multilevelinvertertopologywithpreservedoutputpowerandvoltagelevels vertersunderfaultcondition,’’IEEETrans.Ind.Electron.,vol.56,no.7,
underpre-andpostfaultoperation,’’IEEETrans.Ind.Electron.,vol.68, pp.2697–2703,Jul.2009,doi:10.1109/TIE.2009.2019771.
no.7,pp.5756–5764,Jul.2021,doi:10.1109/TIE.2020.2994880. [85] Z.Wang,X.Yin,andY.Chen,‘‘Anovelnon-redundantsubmodulesfault-
[66] G. Abeynayake, G. Li, T. Joseph, J. Liang, and W. Ming, ‘‘Reliability tolerantcontrolstrategyforMMCinmediumvoltageapplication,’’Int.
andcost-orientedanalysis,comparisonandselectionofmulti-levelMVDC J.Electr.PowerEnergySyst.,vol.139,Jul.2022,Art.no.108048,doi:
converters,’’ IEEE Trans. Power Del., vol. 36, no. 6, pp.3945–3955, 10.1016/j.ijepes.2022.108048.
Dec.2021,doi:10.1109/TPWRD.2021.3051531. [86] J.Song-Manguelle,T.Thurnherr,S.Schroder,A.Rufer,andJ.-M.Nyobe-
[67] P.Tu,S.Yang,andP.Wang,‘‘Reliabilityandcostbasedredundancydesign Yome, ‘‘Re-generative asymmetrical multi-level converter for multi-
for modular multilevel converter,’’ IEEE Trans. Ind. Electron., vol. 66, megawattvariablespeeddrives,’’inProc.IEEEEnergyConvers.Congr.
no.3,pp.2333–2342,Mar.2018,doi:10.1109/TIE.2018.2793263. Expo.,Sep.2010,pp.3683–3690.
VOLUME11,2023 13445

S.Rahimpouretal.:FaultManagementTechniquestoEnhancetheReliabilityofPowerElectronicConverters:AnOverview
[87] H. Truong, C. Mai, C. Nguyen, and P. Vu, ‘‘Modified space vector DMITRIVINNIKOV(Fellow,IEEE)receivedthe
modulationforcascadedH-Bridgemultilevelinverterwithopen-circuit Dipl.Eng., M.Sc., and Dr.Sc.Techn. degrees in
powercells,’’J.Electr.Comput.Eng.,vol.2021,pp.1–14,Mar.2021,doi: electricalengineeringfromtheTallinnUniversity
10.1155/2021/6643589. of Technology, Tallinn, Estonia, in 1999, 2001,
[88] Y. Neyshabouri and H. Iman-Eini, ‘‘A new fault-tolerant strategy for a and2005,respectively.HeiscurrentlytheHead
cascadedH-bridgebasedSTATCOM,’’IEEETrans.Ind.Electron.,vol.65,
|     |     |     |     |     |     |     |     |     | of the | Power | Electronics | Group, | Department | of  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ----- | ----------- | ------ | ---------- | --- |
no.8,pp.6436–6445,Aug.2018,doi:10.1109/TIE.2018.2793182.
|                    |     |               |        |           |            |              |     |     | Electrical | Power | Engineering |     | and Mechatronics, |     |
| ------------------ | --- | ------------- | ------ | --------- | ---------- | ------------ | --- | --- | ---------- | ----- | ----------- | --- | ----------------- | --- |
| [89] M. Aleenejad, |     | H. Iman-Eini, | and S. | Farhangi, | ‘‘Modified | space vector |     |     |            |       |             |     |                   |     |
TallinnUniversityofTechnology,Estonia.Heis
modulationforfault-tolerantoperationofmultilevelcascadedH-bridge
|     |     |     |     |     |     |     |     |     | the | Head of | the Research | and | Development | and |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ----------- | --- |
inverters,’’IETPowerElectron.,vol.6,no.4,pp.742–751,Apr.2013,doi:
10.1049/iet-pel.2012.0543. aCo-FounderofUbikSolutionsLLC—Estonian
start-upcompanydedicatedtoinnovativeandsmartpowerelectronicsfor
[90] P.CorreaandJ.Rodriguez,‘‘Controlstrategyreconfigurationforamulti-
renewableenergysystems.Moreover,heisoneofthefoundersandleading
levelinverteroperatingwithbypassedcells,’’inProc.IEEEInt.Symp.Ind.
|     |     |     |     |     |     |     | researchers | of ZEBE–Estonian |     | Centre | of Excellence |     | for zero energy | and |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---------------- | --- | ------ | ------------- | --- | --------------- | --- |
Electron.,Jun.2007,pp.3162–3167.
resourceefficientsmartbuildingsanddistricts.Hehasauthoredorcoau-
[91] S.Wei,B.Wu,S.Rizzo,andN.Zargari,‘‘Comparisonofcontrolschemes
formultilevelinverterwithfaultycells,’’inProc.30thAnnu.Conf.IEEE thoredtwobooks,fivemonographs,andonebookchapter,aswellasmore
Ind.Electron.Soc.(IECON),vol.2,Nov.2004,pp.1817–1822. than400publishedarticlesonpowerconverterdesignanddevelopmentand
[92] D.Eaton,J.Rama,andP.Hammond,‘‘Neutralshift[fiveyearsofcontin- theholderofnumerouspatentsandutilitymodelsinthisfield.Hisresearch
uousoperationwithadjustablefrequencydrives],’’IEEEInd.Appl.Mag., interestsincludeapplieddesignofpowerelectronicconvertersandcontrol
vol.9,no.6,pp.40–49,Nov.2003. systems, renewable energy conversion systems (photovoltaic and wind),
[93] P.W.Hammond,‘‘Enhancingthereliabilityofmodularmedium-voltage impedance-sourcepowerconverters,andimplementationofwidebandgap
drives,’’IEEETrans.Ind.Electron.,vol.49,no.5,pp.948–954,Oct.2002.
powersemiconductors.HeistheChairoftheIEEEEstoniaSection.
[94] P.W.HammondandM.F.Aiello,‘‘Multiphasepowersupplywithplural
| series | connected | cells and | failed cell | bypass,’’ | U.S. Patent | 5986909, |     |     |       |          |     |           |     |          |
| ------ | --------- | --------- | ----------- | --------- | ----------- | -------- | --- | --- | ----- | -------- | --- | --------- | --- | -------- |
|        |           |           |             |           |             |          |     |     | NASER | VOSOUGHI |     | KURDKANDI |     | (Member, |
Nov.16,1999.
IEEE)wasborninBostanAbad,EastAzerbaijan,
Iran,in1989.HereceivedtheB.Sc.degreeinelec-
tricalengineeringfromtheIslamicAzadUniver-
sity,SouthTehranBranch,Tehran,Iran,in2011,
andtheM.Sc.andPh.D.degreesinelectricalengi-
|     |     |       |           |                      |     |     |     |     | neering | and        | power electronics |       | from the    | Univer- |
| --- | --- | ----- | --------- | -------------------- | --- | --- | --- | --- | ------- | ---------- | ----------------- | ----- | ----------- | ------- |
|     |     |       |           |                      |     |     |     |     | sity    | of Tabriz, | Tabriz,           | Iran, | in 2014 and | 2019,   |
|     |     | SAEED | RAHIMPOUR | (StudentMember,IEEE) |     |     |     |     |         |            |                   |       |             |         |
wasborninZanjan,Iran,in1991.Hereceivedthe respectively.From2019to2020,hewasaPost-
B.S.degreeinelectricalengineeringfromZanjan doctoralResearcherwiththeUniversityofTabriz.
University,Zanjan,in2014,andtheM.S.degree In2020,hejoinedtheTallinnUniversityofTechnology,Tallinn,Estonia,
asaPostdoctoralResearcher.HehasbeenaPostdoctoralResearch-Fellow
|     |     | in electrical | engineering, |     | power | electronics, and |                |       |             |     |        |          |             |     |
| --- | --- | ------------- | ------------ | --- | ----- | ---------------- | -------------- | ----- | ----------- | --- | ------ | -------- | ----------- | --- |
|     |     |               |              |     |       |                  | with San Diego | State | University, | San | Diego, | CA, USA, | since 2022. | His |
electricalmachinesfromGuilanUniversity,Rasht,
|     |     |     |     |     |     |     | current research | interests | include | multilevel | inverters, |     | grid-connected | PV  |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --------- | ------- | ---------- | ---------- | --- | -------------- | --- |
Iran,in2017.HeiscurrentlypursuingthePh.D.
inverters,dc–dcswitched-capacitorandswitched-inductorconverters,fast
degreeinpowerelectronicswiththeTallinnUni-
chargestationforelectricvehicles,battery-basedenergystoragesystems,
versityofTechnology(Taltech),Tallinn,Estonia.
andinductionmotordrives.
|                 |     | His current      | research    | interests | include        | modeling,  |     |     |     |     |     |     |     |     |
| --------------- | --- | ---------------- | ----------- | --------- | -------------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
| control, design | of  | power electronic | converters, |           | fault tolerant | inverters, |     |     |     |     |     |     |     |     |
dcsolid-statecircuitbreakers,energyrouters,andgrid-connectedmultilevel HADITARZAMNI(StudentMember,IEEE)was
inverters.HereceivedtheMostActiveParticipantAwardatRTUCONCon-
|     |     |     |     |     |     |     |     |     | born | in Tabriz, | Iran, | in 1992. | He received | the |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---------- | ----- | -------- | ----------- | --- |
ference,in2021.
|     |     |     |     |     |     |     |     |     | B.Sc.  | and M.Sc.   | degrees | (Hons.) | in power   | elec-      |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ----------- | ------- | ------- | ---------- | ---------- |
|     |     |     |     |     |     |     |     |     | trical | engineering | from    | the     | Faculty of | Electrical |
andComputerEngineering,UniversityofTabriz,
Tabriz,in2014and2016,respectively.Heiscur-
|     |     |     |     |     |     |     |     |     | rently      | pursuing | the         | dual Ph.D. | degree         | in power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ----------- | ---------- | -------------- | -------- |
|     |     |     |     |     |     |     |     |     | electronics |          | engineering | with       | the Department | of       |
ElectricalEngineering,SharifUniversityofTech-
OLEKSANDR HUSEV (Senior Member, IEEE) nology,Tehran,Iran,andtheDepartmentofElec-
received the B.Sc. and M.Sc. degrees in indus- tricalEngineeringandAutomation,AaltoUniversity,Espoo,Finland.Hehas
trialelectronicsfromChernihivStateTechnolog- authored and coauthored more than 30 journal and conference papers.
ical University, Chernihiv, Ukraine, in 2007 and Healsoholdssixpatentsintheareaofpowerelectronics.SinceJanuary2021,
2008,respectively.HedefendedPh.D.thesisfrom he has been a Researcher with the Department of Electrical Engineering
the Institute of Electrodynamics of the National andAutomation;andtheDepartmentofElectronicsandNanoengineering,
Academy of Science of Ukraine, in 2012. He is AaltoUniversity,Finland.Hisresearchinterestsincludepowerelectronic
a Senior Researcher and a Project Leader with convertersanalysisanddesign,DC–DCandDC–ACconverters,highstep-
the Department of Electrical Power Engineering uppowerconversion,soft-switchingandresonantconverters,andreliability
andMechatronics,TalTechUniversity.Hehasover analysis.HewasarecipientoftheBestPaperAwardin10thInternational
100 publications and the holder of several patents. His research interests PowerElectronics,DriveSystemsandTechnologiesConference(PEDSTC),
include in power electronics systems, design of novel topologies, control in2019.Hehasbeenawardedathree-yearAaltoELECDoctoralSchool
systemsbasedonawiderangeofalgorithms,includingmodeling,design, grant,andaJennyandAnttiWihuriFoundationGrant,in2021and2022,
| simulation, | applied | design of power | converters | and | control | systems and | respectively. |     |     |     |     |     |     |     |
| ----------- | ------- | --------------- | ---------- | --- | ------- | ----------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
application,andstabilityinvestigation.
| 13446 |     |     |     |     |     |     |     |     |     |     |     |     | VOLUME11,2023 |     |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- |