| 6332 |             |      |      |       |          |            |     | IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019 |         |         |           |           |     |     |
| ---- | ----------- | ---- | ---- | ----- | -------- | ---------- | --- | ------------------------------------------------------- | ------- | ------- | --------- | --------- | --- | --- |
|      | Reliability |      | of   | Power |          | Converters |     |                                                         |         | in Wind |           | Turbines: |     |     |
|      | Exploratory |      |      |       | Analysis |            | of  | Failure                                                 |         | and     | Operating |           |     |     |
|      |             | Data | From |       | a        | Worldwide  |     |                                                         | Turbine |         |           | Fleet     |     |     |
KatharinaFischer ,KarolinePelka,ArneBartschat ,BerndTegtmeier,DiegoCoronado,
|     |     |     |     |     | ChristianBroer |     |     | ,andJanWenske |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
Abstract—Inviewofthefrequentandcostlyfailuresofpower sites.Itisratherobservedworldwideandinamultitudeofdif-
convertersinwindturbines,alargeconsortiumofresearchinsti- ferent WT types, as proven by numerous studies and widely
tutesandcompanieshasjoinedforcestoinvestigatetheunderlying
|     |     |     |     |     |     |     |     | confirmed | by the | experience | of  | WT operators. | As  | a basis for |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ---------- | --- | ------------- | --- | ----------- |
causesandkeydrivingfactorsofthefailures.Thispaperpresents
|     |     |     |     |     |     |     |     | the development |     | of effective | countermeasures, |     | it  | is crucial to |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------------ | ---------------- | --- | --- | ------------- |
anexploratorystatisticalanalysisofthecomprehensivefielddata
understandtheprevailingmechanismsandcausesofconverter
| provided | by  | the project | partners. | The | evaluated | dataset | covers |     |     |     |     |     |     |     |
| -------- | --- | ----------- | --------- | --- | --------- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- |
converter failures recorded from 2003–2017 during almost 7400 failure in wind turbines. This has been the main subject of a
operatingyearsofvariable-speedwindturbinesofdifferentman- research cluster formed by a consortium of 16 companies in-
| ufacturers |     | and types, operating | at  | onshore | and offshore |     | sites in |                                |     |     |     |                         |     |     |
| ---------- | --- | -------------------- | --- | ------- | ------------ | --- | -------- | ------------------------------ | --- | --- | --- | ----------------------- | --- | --- |
|            |     |                      |     |         |              |     |          | cludingwindturbineandcomponent |     |     |     | manufacturers,operators |     |     |
23countries.Theresultsincludethedistributionoffailureswithin
|     |     |     |     |     |     |     |     | and maintenance-service |     |     | providers, | as well | as Fraunhofer | in- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ---------- | ------- | ------------- | --- |
theconvertersystemandthecomparisonofconverterfailurerates
|     |     |     |     |     |     |     |     | stitutes | and academia | [1], | [2]. | Using a field-experience |     | based |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | ---- | ---- | ------------------------ | --- | ----- |
amongturbineswithdifferentgenerator-converterconcepts,from
differentmanufacturersaswellasfromdifferentturbinegenera- approach,theresearchworkwithinthisclusterbuildsonasys-
tions.Bymeansofcombinedanalysesofconverter-failuredatawith tematic analysis of comprehensive failure and operating data
operatingandclimatedata,conditionspromotingfailureareiden-
|         |          |                  |          |           |         |             |       | provided | by the | project     | partners, | on field     | measurement | cam-        |
| ------- | -------- | ---------------- | -------- | --------- | ------- | ----------- | ----- | -------- | ------ | ----------- | --------- | ------------ | ----------- | ----------- |
| tified. | In line  | with the results | of a     | previous, | much    | smaller     | study |          |        |             |           |              |             |             |
|         |          |                  |          |           |         |             |       | paigns   | and on | post-mortem | analysis  | of converter |             | hardware to |
| of the  | authors, | the present      | analysis | provides  | further | indications |       |          |        |             |           |              |             |             |
identifythemainrootcausesoffailure.
againstthewide-spreadassumptionthatthermal-cyclinginduced
fatigueisthelifetime-limitingmechanisminthepowerconverters
| of wind | turbines. | Instead, | the results | suggest | that | humidity | and |     |     |     |     |     |     |     |
| ------- | --------- | -------- | ----------- | ------- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
A. PowerConvertersinWindTurbines
condensationplayanimportantroleintheemergenceofconverter
failuresinthisapplication.
|     |     |     |     |     |     |     |     | Power | converters | are | key components | of  | almost | all modern |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---------- | --- | -------------- | --- | ------ | ---------- |
Index Terms—Converters, failure analysis, power electronics, windturbines.LocatedbetweenthegeneratorandtheWTtrans-
reliability,windturbine.
|     |     |     |     |     |     |     |     | former, | they enable | variable-speed |     | operation | and | ensure grid- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ----------- | -------------- | --- | --------- | --- | ------------ |
codecompliantbehavioroftheturbine.Thegenerator-converter
I. INTRODUCTION
topologiescommonlyusedinWTare:
THE
main power-converter system in wind turbines (WT) 1) doubly-fed induction generators (DFIG) with partially
isafrequentsourceoffailureand,assuch,adriverofre- ratedconverters;
paircostanddowntime.Theproblemofhighconverterfailure
|     |     |     |     |     |     |     |     | 2) squirrel-cageinductiongeneratorswithfull-scaleconvert- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
ers(IG+FPC);
| rates | is limited | neither to | single | manufacturers |     | nor to specific |     |                     |     |     |                 |         |     |             |
| ----- | ---------- | ---------- | ------ | ------------- | --- | --------------- | --- | ------------------- | --- | --- | --------------- | ------- | --- | ----------- |
|       |            |            |        |               |     |                 |     | 3) permanent-magnet |     |     | or electrically | excited |     | synchronous |
generators(PMSG,EESG)withfullyratedconverters.
ManuscriptreceivedMarch19,2018;revisedJune12,2018andAugust20,
Theconvertertechnologyprevailinginthewindapplication
2018;acceptedSeptember24,2018.DateofpublicationOctober10,2018;date
|     |     |     |     |     |     |     |     | consists | of low-voltage |     | two-level | IGBT-based | voltage | source |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------------- | --- | --------- | ---------- | ------- | ------ |
ofcurrentversionMay2,2019.ThisworkwassupportedinpartbytheGerman
FederalStateofLowerSaxonywithfundsfrom“Niedersa¨chsischesVorab”and converters in back-to-back configuration. Fig. 1 shows such a
inpartbyFraunhofer-Gesellschaft.RecommendedforpublicationbyAssociate
converterinitssimplestform.
EditorA.K.Gupta.(Correspondingauthor:KatharinaFischer.)
K. Fischer, K. Pelka, A. Bartschat, B. Tegtmeier, and C. Broer are with Duetothelimitedcurrent-carryingcapacityofasinglepower
the Fraunhofer Institute for Wind Energy Systems (Fraunhofer IWES), semiconductor,eachdeviceinFig.1representsalargernumber
Hannover D-30167, Germany (e-mail:,katharina.fischer@iwes.fraunhofer.de; ofparalleledIGBTsordiodes,respectively.Ifnecessary,several
| karoline.pelka@iwes.fraunhofer.de; |     |     |     | arne.bartschat@iwes.fraunhofer.de; |     |     |     |     |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bernd.tegtmeier@iwes.fraunhofer.de;christian.broer@iwes.fraunhofer.de). oftheunitsshowninthisfigureareagainconnectedinparallel
D.CoronadowaswiththeFraunhoferInstituteforWindEnergySystems, inaWT,typicallywithajointDClink.
HannoverD-30167,Germany.HeisnowwithCGNEuropeEnergy,ParisF-
92977,France(e-mail:,diego.coronado@cgnpc.com.cn). Three-level and modular multilevel converters for the wind
J. Wenske is with the Fraunhofer Institute for Wind Energy Systems, application are commercially available, but have not achieved
BremerhavenD-27572,Germany(e-mail:,jan.wenske@iwes.fraunhofer.de). a broad distribution in the field yet. Generally speaking, the
Colorversionsofoneormoreofthefiguresinthispaperareavailableonline
widelyexpectedshifttowardtheuseofmedium-voltage(MV)
athttp://ieeexplore.ieee.org.
DigitalObjectIdentifier10.1109/TPEL.2018.2875005 convertersinWTwithmorethan3–5MWisnotobservedsofar.
0885-8993©2018IEEE.Personaluseispermitted,butrepublication/redistributionrequiresIEEEpermission.
Seehttp://www.ieee.org/publicationsstandards/publications/rights/index.htmlformoreinformation.
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

FISCHERetal.:RELIABILITYOFPOWERCONVERTERSINWINDTURBINES:EXPLORATORYANALYSISOFFAILUREANDOPERATINGDATA 6333
|     |     |     |     |     | reliability | study     | based       | on >5800 | failures      | reported | during       |
| --- | --- | --- | --- | --- | ----------- | --------- | ----------- | -------- | ------------- | -------- | ------------ |
|     |     |     |     |     | 2010–2012   | in a      | fleet of    | Chinese  | wind turbines | is       | published in |
|     |     |     |     |     | [14]. In    | line with | the results | of       | European      | studies, | the largest  |
portionoffailuresisallottedtothepowerconverteralsointhe
Chinesefleet.
Thewind-powerspecificscientificliteraturedealingwithlife-
|     |     |     |     |     | time estimations |          | of power | converters    | that    | falls into | the above   |
| --- | --- | --- | --- | --- | ---------------- | -------- | -------- | ------------- | ------- | ---------- | ----------- |
|     |     |     |     |     | named            | category | (b) has  | been strongly | focused | on         | the failure |
mechanismsofpowerelectronicsexperiencedinotherapplica-
tionssofar:thegradualdegradationofdie-attachsolderaswell
|     |     |     |     |     | as the lift-off | or  | damage | of bond | wires | (see e.g., | [15]–[20]). |
| --- | --- | --- | --- | --- | --------------- | --- | ------ | ------- | ----- | ---------- | ----------- |
Bothfailuremodesresultfromfatiguecausedbythelong-term
|     |     |     |     |     | impact | of thermal | cycling | of material | combinations |     | with dif- |
| --- | --- | --- | --- | --- | ------ | ---------- | ------- | ----------- | ------------ | --- | --------- |
ferentthermalexpansionbehavior.Forthese“classical”failure
Fig.1. Voltage-sourceconvertertopologyprevailinginwindturbines. modesofpowerelectronics,lifetimemodelsareavailable(e.g.,
[21]–[23]),whichmightexplaintheattractivenessofassuming
|     |     |     |     |     | their dominance |     | for theoretical |     | lifetime calculations. |     | Further- |
| --- | --- | --- | --- | --- | --------------- | --- | --------------- | --- | ---------------------- | --- | -------- |
Asfirstexamples,theturbinetypesArevaM5000/AdwenAD5,
Mervento3.6,XEMCXD115,andAdwenAD8canbenamed, more, also the design and accelerated life testing of today’s
allofwhichareequippedwithMV-PMSGandfullyratedcon- power modules are primarily focused on these fatigue-based
|                    |        |               |              |        | failure mechanisms. |     | One | of the | consequences | of  | this strong |
| ------------------ | ------ | ------------- | ------------ | ------ | ------------------- | --- | --- | ------ | ------------ | --- | ----------- |
| verters. Depending | on the | manufacturer, | either IGCTs | (e.g., |                     |     |     |        |              |     |             |
ABB)orMV-IGBTs(e.g.,Converteam,Ingeteam)areapplied focus on bond-wire and solder fatigue is that the wind-power
in the MV converters [3]. However, the majority of new tur- specificliteraturehasconcentratedmainlyonthemachine-side
binetypes,suchasSiemensGamesaSWT-7.0/SG-8.0,Enercon converter in WT with DFIG (see e.g., [24]–[28]) as well as
E126-7.58, and MHI Vestas V164-8.0, continue to use low- in WT with low-speed PMSG (e.g., [29]–[32]). Because these
|              |             |                     |          |     | converters | are subject | to  | high currents | at  | low frequencies | and |
| ------------ | ----------- | ------------------- | -------- | --- | ---------- | ----------- | --- | ------------- | --- | --------------- | --- |
| voltage IGBT | converters. | For a comprehensive | overview | of  |            |             |     |               |     |                 |     |
thedifferentpower-electronicconvertersappliedinorsuitable thereforetoparticularlyseverethermalcycling[33],theyhave
forWT,includinglow-voltageandmedium-voltagetopologies, beenconsideredtohaveaparticularlyhighriskoffailure.
pleasereferto[4]. WithintheprojectCONFAIL[34],[35],converterfailurein
WTwassystematicallyinvestigatedforthefirsttimebymeans
|     |     |     |     |     | of a field-data | and | field-experience |     | based | approach, | with the |
| --- | --- | --- | --- | --- | --------------- | --- | ---------------- | --- | ----- | --------- | -------- |
B. ResearchonPowerConverterReliabilityinWindTurbines
|     |     |     |     |     | objective | to shed | light on | the causes | of failure. | Subject | of the |
| --- | --- | --- | --- | --- | --------- | ------- | -------- | ---------- | ----------- | ------- | ------ |
Thepubliclyavailablescientificliteraturecoveringtherelia- investigationweretwoWTtypesofsimilarratedcapacity,but
bilityofthe(main)convertersinWTcanbemainlyassignedto fromdifferentmanufacturers,oneofthemwithDFIG,theother
IG+FPC-type.
| twocategories: |     |     |     |     | of  |     | In CONFAIL, |     | field data | from | approx. 370 |
| -------------- | --- | --- | --- | --- | --- | --- | ----------- | --- | ---------- | ---- | ----------- |
1) field-databasedfailurestatisticsorreliabilitystudies,re- operating years were evaluated. The analysis of this data as
spectively,typicallywithasystem-levelfocus; wellasofseveraldefectpowermodulesfromthefieldprovided
2) theoreticallifetimecalculationbasedonmissionprofiles indications of insufficient protection of the converters against
andexistinglifetimemodels,however,withoutconsider- the environment (salt, insects, humidity, and condensation) as
ationofthefailuremodesoccurringinthefield. wellas,incaseofonewindpark,sometemporalcoincidenceof
Majorcontributionsinthecategory(a)includethereliability thunderstormsandconverterfailures.Aparticularlyinteresting
study based on 373 variable-speed WT with rated capacities resultwasthatthepost-mortem-analyzedpowermoduleslacked
(cid:2)850 kW carried out within the RELIAWIND project, using anysignsoffatiguedamage—anobservationthathasraisedthe
datafrom2004–2010coveringintotal1115WToperatingyears questionifthermal-cyclinginducedfatiguedamageplaysinfact
[5],[6].Inaddition,thecomprehensiveworkattheUniversity arelevantroleintheemergenceoftheconverterfailuresinWT.
of Durham (e.g., [7], [8]) on the basis of field data from Toanswerthisquestionbasedonamuchmorecomprehensive
1993–2004,aswellasthatofISET/FraunhoferIWESbasedon and varied WT fleet and to identify the (possibly different)
datafromtheWMEPprogramfrom1989–2006(e.g.,[9],[10]) key drivers of converter failure in the field is among the main
fall into this first category. Their relevance for today’s wind objectives of the initially introduced research cluster and the
turbinetechnologyis,however,limitedbytheoldandtechno- workpresentedinthefollowing.
logically in many cases outdated WT fleets underlying these Apreviouspaperoftheauthors,[36],presentedfirstresults
studies. More recent reliability studies on WT have been pub- ofstatisticalanalysescarriedoutwithinthisresearchcluster.It
lishedbytheUniversityofStrathclyde(cf.[11]–[13]).Theseare usedadatasubsetcovering1269wind-turbineoperatingyears
basedoncomprehensivefielddatafromWTofoneofthelead- toidentifythemostfrequentlyfailingcomponentsandthemain
ingmanufacturers.However,asthemainpurposeofthatworkis repair-costdriverswithinthepower-convertersystem.
operationsandmaintenance(O&M)modeling,itprovideslittle Thispaperpresentsconverter-specificfailurestatisticsbased
informationthatcouldsupportanidentificationofthecausesof on a substantially extended field-data basis provided by the
power-converterfailure.Anotherrelativelyrecentsystem-level partners of the research cluster, which now includes almost
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

| 6334 |     |     |     |     |     |     | IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019 |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Fig.2. Distributionoftheevaluateddiversewind-turbinefleetovercontinents
(portionsbasedonthenumberofturbines).
| 7400 years | of wind-turbine | operation. |     | The work | is novel | and |     |     |     |     |     |     |     |     |
| ---------- | --------------- | ---------- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
uniquedueto
|     |     |     |     |     |     |     | Fig.3. Amountofevaluateddatafromwindturbinesofdifferentages(bar |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
1) thecomprehensiveness,depth,andtimelinessofthedata
chart)andportionsofturbinesbydifferentmanufacturers(piechart,basedon
| basis; |     |     |     |     |     |     | evaluatednumberofWToperatingyears). |     |     |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- |
2) thefocusonstatisticalanalysisforthepurposeoffailure
root-causeidentification;
|        |                     |     |            |     |           |       | Inside the  | turbines, | the       | power | converters     | are | located | inside |
| ------ | ------------------- | --- | ---------- | --- | --------- | ----- | ----------- | --------- | --------- | ----- | -------------- | --- | ------- | ------ |
| 3) the | combined evaluation |     | of failure | and | operating | data, |             |           |           |       |                |     |         |        |
|        |                     |     |            |     |           |       | the nacelle | or in     | the tower | base. | Alternatively, |     | in some | WT     |
whichallowstoidentifyoperatingconditionspromoting
|     |     |     |     |     |     |     | types, the | machine-side | converter |     | is placed | in  | the nacelle | and |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------------ | --------- | --- | --------- | --- | ----------- | --- |
converterfailure.
thegrid-sideconverterinthetowerbase.InsinglecasesofWT
|     |     |     |     |     |     |     | built on lattice | towers, | the | converters | are | located | in a | station |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ------- | --- | ---------- | --- | ------- | ---- | ------- |
C. PaperOutline next to the tower base. The converters are either air-cooled or
liquid-cooled.
Thesubsequentpartsofthepaperarestructuredasfollows:
Thedataincludesconverterfailureeventsfrom2003to2017.
| Section II | describes the | data basis | underlying |     | the subsequent |     |            |        |          |      |       |         |          |      |
| ---------- | ------------- | ---------- | ---------- | --- | -------------- | --- | ---------- | ------ | -------- | ---- | ----- | ------- | -------- | ---- |
|            |               |            |            |     |                |     | The length | of the | periods, | from | which | failure | data has | been |
analysisandintroducestheprocessingandevaluationmethods.
evaluated,rangesfromseveralmonthsto12yearsperturbine.
| Section III | presents and | discusses | the | results | of the | different |                    |     |       |        |              |     |       |         |
| ----------- | ------------ | --------- | --- | ------- | ------ | --------- | ------------------ | --- | ----- | ------ | ------------ | --- | ----- | ------- |
|             |              |           |     |         |        |           | Fig. 3 illustrates | the | total | number | of operating |     | years | covered |
dataanalyses.SectionIVsummarizesthemainconclusionsand
|     |     |     |     |     |     |     | by the dataset | from | turbines | of  | different | ages. | Note | that this |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ---- | -------- | --- | --------- | ----- | ---- | --------- |
providesanoutlooktofurtherwork.
includesbothturbineswithandwithoutconverterfailures.With
atotalnumberofmorethan1500turbineoperatingyears,most
dataareavailablefromturbinesintheirfirstyearofoperation.
| II. | DATABASISANDMETHODSOFANALYSIS |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thedatasetincludessimilaramountsofdatafromturbineages
A. DataUnderlyingtheAnalysis
of3–13yearsand—toaminorextent—fromolderturbineswith
The failure data covers in total 2734 wind turbines with a anageofupto19years.Intotal,theevaluatedfailuredatacovers
main converter, which are operating in 23 different countries almost7400WToperatingyears,withaportionofapprox.10%
spreadoverfourdifferentcontinents(seeFig.2). fromoffshorewindparks.Forasmallportionoftheevaluated
TheturbinesareequippedwithDFIG,EESG,orIG+FPC.In
|     |     |     |     |     |     |     | fleet, the turbine |     | age is | unknown. | The | pie chart | included | in  |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ------ | -------- | --- | --------- | -------- | --- |
addition,therearesingleturbineswithPMSG.Thedatasetcov- Fig. 3 shows the data portions corresponding to the each of
ers turbines by 11 different manufacturers, namely DeWind, the 11 WT manufacturers (OEM). In addition to the failure
Enercon, Fuhrla¨nder, Gamesa, General Electric, Kenersys, data, the data basis comprises operating histories in the form
Nordex, Senvion, Suzlon, Siemens, and Vestas. The nominal of 10 min-averaged signals from the Supervisory Control and
poweroftheturbinesrangesfrom500to3600kW.Theircom- Data Acquisition (SCADA) system as well as SCADA status-
missioningdatesspreadovertheyears1997–2015. logsfrommorethanhalfoftheturbines.
All power converters in the investigated turbines are low- The converter-specific failure data has been derived from
voltage two-level IGBT-based voltage source converters, i.e., maintenancereportsincludinginformationonusedspareparts
of the prevailing technology in wind turbines (cf. Section I- or from turbine logbooks. Only faults requiring on-site repair
A). In the EESG-based WT, the AC–DC conversion on the and the consumption of material or spare parts are counted as
generatorsideisimplementedbymeansofdiode-orthyristor- failure events. Faults remedied by measures such as a remote
based rectifiers. In addition, these turbines contain a unit for reset,cleaning,orretighteningofcomponentsarethereforenot
| theelectricalexcitationofthegeneratorand,dependingonthe |     |     |     |     |     |     | included. |     |     |     |     |     |     |     |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
WT type, a step-up unit in the DC link. In the present paper, Based on the above information, failures are classified us-
thesecomponentsareconsideredaspartsofthepower-converter ingthefollowingconverter-componentcategories:phasemod-
system. ule(includingIGBTmodulesandcorrespondingdriverboards,
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

FISCHERetal.:RELIABILITYOFPOWERCONVERTERSINWINDTURBINES:EXPLORATORYANALYSISOFFAILUREANDOPERATINGDATA 6335
DC-linkcapacitors,busbars;inEESG-basedWTalsothecorre-
spondingcomponentsintheexcitationunit),convertercontrol
board,coolingsystem,maincircuitbreaker,grid-couplingcon-
| tactor, and | other        | converter | failures.  | Note     | that a          | failure    | event |     |     |     |     |     |     |     |
| ----------- | ------------ | --------- | ---------- | -------- | --------------- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- |
| can cover   | several      | of these  | categories | in       | case components |            | of    |     |     |     |     |     |     |     |
| more than   | one category |           | were       | replaced | in order        | to restore | the   |     |     |     |     |     |     |     |
functionalityoftheconverter.
B. CalculationofFailureRatesandTheir
ConfidenceIntervals
Thepresentanalysisisbasedonaveragefailurerates,which
isacommonmeasureforassessingthereliabilityofacompo-
| nent, not | at least | due to | its suitability |     | for cases | with lacking |     |     |     |     |     |     |     |     |
| --------- | -------- | ------ | --------------- | --- | --------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
informationaboutthecomponentage.Theaveragefailurerate
ofacomponentisdefinedas
(cid:2)
I N N Fig.4. Identificationofoperatingconditionsprecedingconverterfailurefrom
i
f = (cid:2) i=1 = (1) theoperatinghistory(SCADA10mindata)ofa1.5MWwindturbine.
|     |     |     | I X | ·T  | T   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i i
i=1
withN idenotingthenumberoffailureeventsofthiscomponent
inthetimeintervali,X fortheaveragefailurerateincorporatingtime-censoringisesti-
i describingthetotalnumberofturbines
evaluatedinthistimeinterval,andT beingthedurationofthe matedaccordingto
i
|     |     |     |     |     |     |     |     |     | (cid:3) (cid:4) | (cid:5) | (cid:4) |     | (cid:5)(cid:6) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- | ------- | ------- | --- | -------------- | --- |
timeinterval.Theoverallnumberofobservedfailureeventsof χ2 α, χ2 α,
|     |     |     |     |     |     |     |     |     |     | 2N  | 1−  | 2N  | +2  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the component of interest will be denoted as N and the total 2 , 2 (2)
|     |     |     |     |     |     |     |     |     | 2T  |     |     | 2T  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
numberofevaluatedwind-turbineoperatingyearsasT,includ-
ingbothfailedandnonfailedturbines.Allfailureratesaregiven whereχ2(α/2,2N)isthe(α/2)-quantileofthechi-squaredistri-
| infailuresperwindturbineandyear(1a−1(cid:3)1.14·105 |                     |         |                    |     |             | FIT). |         |         |                 |               |     |          |         |            |
| --------------------------------------------------- | ------------------- | ------- | ------------------ | --- | ----------- | ----- | ------- | ------- | --------------- | ------------- | --- | -------- | ------- | ---------- |
|                                                     |                     |         |                    |     |             |       |         | bution  | with 2N degrees | of freedom.   |     | In the   | present | paper, the |
| Although                                            | average             | failure | rates              | are | widely used | in    | wind-   |         |                 |               |     |          |         |            |
|                                                     |                     |         |                    |     |             |       |         | average | failure rates   | are presented |     | with the | 90%     | confidence |
| turbine                                             | failure statistics, |         | the conclusiveness |     | of the      | sole  | failure |         |                 |               |     |          |         |            |
intervals,correspondingtoα=0.1.
ratesislimitedbythefactthattheydonotprovideinformation
|     |     |     |     |     |     |     |     | For the | abovementioned |     | example, | this | results in | a confi- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------------- | --- | -------- | ---- | ---------- | -------- |
abouttheuncertaintytheirvalueisafflictedwith.Forexample, [0.1085 a−1, a−1]
|     |     |     |     |     |     |     |     | dence interval | of  |     | 0.3392 |     | for the | case of |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | ------ | --- | ------- | ------- |
theoccurrenceof10failuresduring50WToperatingyearsre-
10failuresobservedin50WToperatingyears.Incontrastwith
sultsinthesamefailurerateof0.2a−1perturbineas100failures
[0.1683a−1,0.2362a−1],itismorenarrowincaseof100failures
observedinatotalof500operatingyears,whiletheuncertainty
during500WToperatingyears,indicatingahighercertaintyof
ofthefailure-ratevalueismuchhigherinthefirstcase.Asthe
thecalculatedaveragefailurerateinthesecondcase.
presentpaperwillcomparethefailureratesofdifferentgroups
ofturbines,withlargedifferencesinthecoveredamountoftur-
bineoperatingyearsbetweenthegroups,ittakesastepbeyond C. CombinedEvaluationofFailureandOperatingData
previousfailure-rateanalysesbyquantifyingtheuncertaintyof Forthefailureeventsofselectedconverter-componentcate-
theprovidedfailureratesbymeansofconfidenceintervals. gories,thedateandtimeoffailureaswellastheoperatingcondi-
Instead of a single value, a confidence interval provides a tionsinthe10min-intervalprecedingthefailuresareidentified
range for a parameter of a population’s probability distribu- fromtheturbines’SCADA10min-dataand/orstatus-logs.Fig.4
tion. Its lower and upper limit are constructed to meet a level illustratesthisprocedure.Itshowsa3-dayexcerptofthe10min-
|     |     | − α), |     | < α < |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
of confidence (1 with 0 1. A two-sided confi- averagedsignalswindspeed,activepower,andgeneratorspeed,
denceintervalfortheparameterθwithconfidencelevel(1−α) withalow-windperiodonthefirstdayandpart-loadoperation
|     |     |     |     | [θ , | θ ] |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
is an interval with the boundaries l u that has the prop- endedbyafailureeventonthesecondday.Theinformationthat
erty P(θ l ≤θ ≤θ u )=1−α. When confidence intervals are thefailureconcernsacomponentofthepower-convertersystem
estimated based on sample data, the confidence level is to be isobtainedfromthemaintenancerecordsandthestatus-logs.
interpretedintermsoffrequency:Ifalargenumberofsamples A magnified view of the signals around the time of failure
(i.e.,failuredatasetsfromagivenpopulationofwindturbines) inFig.4showsthesuddendropofthegeneratorspeedandthe
wasevaluated,theconfidencelevelwouldbeequaltothepro- active power while the wind speed is still well above the tur-
portionofthesampleswhoseconfidencelimitscontainthetrue bine’scut-inwindspeed.Itisimportanttonotethatthelast10
parameter. min-averagedvalueabovezerobeforethedowntimeistypically
T
The time periods i in the dataset evaluated for the present notasuitablecharacterizationoftheturbineoperatingcondition
paperaretime-censored,i.e.,theyarenotterminatedwithcom- inthemomentoffailure,becauseitistheresultofanaveraging
ponent failure. As described in [37], the confidence interval overa10min-intervalthatcoversbothoperationanddowntime.
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

6336 IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019
Fig.5. Distributionoffailedconvertercomponentsovercategories,basedon
3829operatingyearsofwindturbinesofdifferentgenerator-converterconcepts
andtypes.
Fig.6. Comparisonofaveragefailureratesoftheoverallconvertersystem(left
To identify the approximate load range in which the converter bar)andthephase-modulecomponents(rightbar)asthecoreoftheconverter
failureoccurred,itisthereforenecessarytoevaluatethepreced- amongwind-turbinegroupswithdifferentgenerator-converterconcepts.
ing time-step. In the case shown in Fig. 4, this corresponds to
thelocalmaximumofthegeneratorspeedandtheactivepower available. Solely in the group of turbines with EESG and full
of 342 kW. As the turbine has a rated power of 1500 kW, this power converters, a further distinction of failed components
failure event is assigned to a load range of 20%–30% of the within the phase-module category has been possible. In these
ratedpower. turbines, power-module and driver-board failures occurred at
similar frequencies and prevailed clearly over failures of DC
linkcapacitorsorbusbars.
III. RESULTSANDDISCUSSION
Based on the field data described in the previous section, a
B. ConverterFailureRatesofWindTurbinesWithDifferent
varietyofdifferentanalyseshavebeencarriedout.Theirresults
Generator-ConverterConcepts
arepresentedanddiscussedinthefollowing.
Differentgeneratortypesinvolvedistinctiveoperatingcondi-
tionsandloadsfortheconverter.Insearchoffactorsbenefitting
A. DistributionofFailuresOverConverterComponents
orcounteractingconverterreliability,itisthereforeareasonable
Inordertoidentifythepredominantlyfailingcomponentsin steptocomparetheconverterfailureratesofWTwithdifferent
theconvertersofWT,allturbinesforwhichfailuredatafromthe generator-converterconcepts.
completeconvertersystemisavailablehavebeenevaluatedwith Fig. 6 shows the average converter failure rates of turbines
respecttothedistributionoftheaffectedconvertercomponents. with DFIG, EESG, and IG+FPC as well as of the evaluated
Fig. 5 shows the distribution of failed components over the turbine fleet in total. In addition to the bars indicating the av-
converter-componentcategoriesdefinedinSectionII-A. erage failure rates, the diagram includes the confidence inter-
Takingintoaccountthatthecategory“Other”coversalarge vals calculated according to (2) in Section II-B. Note that the
numberofdifferent—mostlyminor—items,itrevealsthatwith amountofavailabledata,characterizedbythesumofevaluated
22% the phase-module category makes up the largest portion WT operating years, varies considerably between the groups
offailedconvertercomponents.In26%ofallconverter-failure of turbines, but also within the groups of turbines. The for-
events,thephase-modulecategorywasaffected.(Notethatthe mer results from the composition of the turbine fleet covered
discrepancybetween22%and26%iscausedbythefactthatin bythedata(seeSectionII-A),whichisstronglydominatedby
a part of the failure events components from more than one DFIGturbines.Thelatterisaconsequenceofthefactthatonly
category were found to be defect, so that the overall num- phase-modulefailureswerereportedfromaportionofthetur-
ber of failed components is larger than the number of failure bines. In case of the turbine group “IG+FPC,” data covering
events.)Besidesthephasemodule,thecoolingsystem(mainly the complete converter system was considered too limited to
with fan and pump failure as well as coolant leakage) and the allow conclusive results and was therefore excluded from the
control board of the converter are frequently failure-affected analysis.
components.Asthephase-modulecategorystandsoutnotonly On average, 0.48 power-converter failures per wind turbine
with respect to the portion of failures, but also with a high re- andyearoccurredinthefleet.Amongtheseare0.16a−1phase-
paircostamongtheconvertercomponents,asapreviousstudy modulefailuresperWT.AsindicatedbyFig.6,thefailurerates
of the authors [36] showed, a particular focus will be set on obtained for the fleets with DFIG, EESG, and IG+FPC differ
theanalysisofphase-modulefailuresinthefollowingpartsof strongly:TheEESGfleetshowsasignificantlylowerfailurerate
this paper. In the majority of the WT investigated, the phase thantheDFIGfleet,bothregardingthephase-modulecategory
modules are replaced as a unit after a failure and information andregardingthecompleteconvertersystem.Theturbineswith
aboutthelocationofthedefectwithinthephasemoduleisnot IG+FPChavethehighestphase-modulefailurerate.
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore. Restrictions apply.

FISCHERetal.:RELIABILITYOFPOWERCONVERTERSINWINDTURBINES:EXPLORATORYANALYSISOFFAILUREANDOPERATINGDATA 6337
Fig.7. AveragefailureratesoftheoverallconvertersystemanditscomponentsingroupsofDFIG-basedwindturbinesofmanufacturersOEM1...OEM8.
The widths of the failure-rate confidence intervals in Fig. 6 to0.266a−1(OEM4)amongthefleetsfromdifferentmanufac-
dependonboththenumberoffailuresandtheamountofeval- turers, i.e., by a factor of almost eight. The results make clear
uatedoperatingyears,cf.(2).ThisisclearlyreflectedinFig.6, that, in spite of the widespread occurrence of high converter
where the failure rates calculated from a large data basis such failure rates in wind turbines, low converter reliability is not
asincaseoftheDFIGturbineshaveanarrowconfidenceinter- inherenttothewindapplication;thefactthattheconvertersand
val(e.g.,[0.149a−1,0.164a−1]forthephase-modulefailures) in particular the phase modules in WT of certain manufactur-
while the uncertainty is much higher in case of the relatively ersachievesuchlowfailure-ratelevelsprovesthatarelatively
smallfleet ofturbines withIG+FPC(withabroad confidence high reliability can be realized already with existing converter
intervalof[0.293a−1,0.443a−1]). technology,whichisanimportantfinding.
In addition to the different reliability levels, Fig. 7 shows
that the converter systems used by the WT manufacturers
C. ConverterFailureRatesofDFIG-BasedWindTurbines
OEM1...OEM8 have different weak points: In most of the
ofDifferentManufacturers
cases, the phase-module category has the highest failure rate
TurbineswithDFIGconstitutethelargestportioninthedata (apart from the cumulative category “Other”). In the case of
basisassembledforthepresentwork.Thisallowssplittingthe singlemanufacturer,however,thefailureratesoftheconverter
DFIG-based fleet into sub-groups of WT in order to compare cooling system (OEM1 and 7) or of the main circuit breaker
theconverterreliabilityinDFIGturbinesofdifferentmanufac- and the converter control-board (OEM1) exceed those of the
turers.Fig.7showstheresultingdiagram.ThenamesoftheWT phase-modulecategoryconsiderably.
manufacturers are anonymized for the sake of confidentiality.
Theevaluationisbasedonfailuredatafrom75to2316WTop-
D. Generator-SideVersusGrid-SideConverterFailures
eratingyearsperOEM(cf.Fig.3).BesidestheWTbymanufac-
inDFIGTurbines
turerOEM6,forwhichonlyphase-modulefailuredatahasbeen
available, the diagram shows the failure rates of all converter- Inadditiontothefailureratesofthedifferentconvertercom-
componentcategories.Notethatthefailureratesofthedifferent ponentspresentedintheprevioussections,alsothedistribution
component categories do not sum up to the converter-system of failures over the machine-side and the line-side converter
failurerate,butthattheirsumisusuallyhigherthantheoverall can reveal relevant insights. This applies in particular for tur-
failurerate.Thisresultsfromthefactthatapartofthefailures bineswithDFIG:IfthelifetimeofconvertersinWTwasindeed
has affected more than one component category. Such cases predominantly determined by thermal cycles and the resulting
are included in the failure rate of each affected category, but fatigue of bond-wire connections and die-attach solder on the
considered only a single failure event in the calculation of the chiplevel,asitiswidelypostulatedinthescientificliterature,
converter-systemfailurerate. failures should be observed mainly in the machine-side con-
Fig. 7 reveals that there are large differences in the failure verterinDFIGturbines,whichissubjecttoparticularlysevere
behavior of the turbines of different manufacturers, both with thermalcyclingduetothelowcurrentfrequencies.
respect to the reliability level and the distribution of failures This is investigated on the basis of the field data described
withintheconvertersystem.Theaveragefailurerateoftheover- above. The analysis can be carried out for DFIG turbines of
allconvertersystemrangesfrom0.235a−1perturbine(OEM5) three different manufacturers, namely OEM3, 5, and 6 intro-
toavalueashighas1.539a−1 (OEM1).Onasimilarnote,the ducedinSectionIII-C,withintotal660failureeventsofIGBT
phase-modulefailureratesvaryfrom0.034a−1perWT(OEM7) modules(incl.thedriverboard).IncaseofOEM3andOEM6,
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore. Restrictions apply.

| 6338 |     |     |     |     |     |     | IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019 |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Fig.8. DistributionofIGBT-modulefailuresovermachine-sideconverter(MSC),line-sideconverter(LSC)andchopperinDFIG-basedwindturbines.
theevaluatedfailuredataoriginatefromthefirstfiveyearsafter However, the drawback of this procedure is that the group of
commissioning, while in case of OEM5 the analysis includes newestWTcontainsonlydatafromthefirstyearsofoperation
datafromWTwithoperatingagesupto13years. (corresponding to the left-side part of Fig. 3) while the group
Fig.8showsthedistributionofIGBT-modulefailuresoverthe of oldest turbines commissioned until 2005 covers data from
machine-side converter (MSC), the line-side converter (LSC), the full range of WT ages and with that also from all phases
aswellasthechoppermoduleusedforvoltagelimitationinthe of the so-called “bathtub curve,” i.e., early-failure, intrinsic-
DC link. In case of OEM3, MSC, and LSC failures are found failure, and deterioration behavior. As according to [8], [11],
inapproximatelyequalportions,whileLSCclearlyoutnumber and [38], converters display early-failure characteristics, there
MSCfailuresintheWTofmanufacturerOEM6.Thisisastrong isariskthatthedifferentturbine-agerangeinthethreegroups
indicationagainsttheassumptionthatthermal-cyclinginduced systematicallyinfluencestheresult.
fatigue is a relevant cause of the observed converter failures. Therefore, in a second analysis step, only data from the
SolelyincaseofOEM5,thelargestportionoffailuresisfound third to sixth operating year of each WT has been evalu-
in the MSC. The fact that, in contrast to the other two cases, ated (cf. the lighter bars in Fig. 9), in order to guarantee a
also older WT of operating ages up to 13 years are included comparability of the failure-rate results. This procedure, in
raisesthequestioniffatiguedamagemightplayaroleatleastin turn, has the disadvantage that it considerably reduces the
thisgroupwitholderturbines.However,adetailedanalysisof amountofevaluateddata,whichleadstowiderconfidencein-
severalfield-returnedIGBTmodulesfromtheseoldestWThas tervals for the failure rates compared to the former case; for
revealed no indications of fatigue damage: The attachment of the group of EESG-based turbines commissioned 2011–2015,
thebond-wirestothechipsaswellastheinvestigateddie-attach the remaining dataset becomes even insufficient for statistical
| solder layers | were | found | in a good | condition, |     | which makes | evaluation. |     |     |     |     |     |     |
| ------------- | ---- | ----- | --------- | ---------- | --- | ----------- | ----------- | --- | --- | --- | --- | --- | --- |
thermal-cyclinginducedfatigueanunlikelycauseofconverter Fig.9givesaclearanswertotheinitialquestion:Converter
failuresalsoinWTofmanufacturerOEM5. failure is not a problem limited to older wind-turbine gener-
|     |     |     |     |     |     |     | ations. Even | if the phase-module |           | failure | rate    | of the      | newest |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------------- | --------- | ------- | ------- | ----------- | ------ |
|     |     |     |     |     |     |     | EESG-based   | turbines is         | afflicted | with    | a large | uncertainty | and    |
E. ConverterFailure—AProblemofAgingFleets? should therefore be considered with caution, it can be stated
thattheconverterreliabilityofcontemporaryWTisnothigher
| An important | question | in  | the context | of  | this work | is if high |     |     |     |     |     |     |     |
| ------------ | -------- | --- | ----------- | --- | --------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
thanthatofWTcommissioned10–15yearsearlier.Asamatter
converterfailureratesareanissueofolderwind-turbinegener-
ofcourse,itshouldnotbeignoredthattheaverageratedpower
ationsonlyorifthiscontinuestobeaproblemincontemporary
ofconvertersinWThasincreasedduringthistime,sothatthere
turbines.Toshedlightonthisquestion,theturbinepopulations
|     |     |     |     |     |     |     | are on average | less failures | per | produced | kWh. | On  | the other |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ------------- | --- | -------- | ---- | --- | --------- |
coveredbythegivendatabasishavebeenseparatedintothree
hand,however,largerconvertersinvolvemoreexpensivespare
| groups: wind | turbines | commissioned |     | 1) until | 2005; | 2) during |     |     |     |     |     |     |     |
| ------------ | -------- | ------------ | --- | -------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- |
partsandlogisticsincaseoffailureandcausehigherproduction
2006–2010;and3)during2011–2015.Turbineswithunknown
(i.e.,revenue)lossesduringWTdowntime.Theresultsobtained
| year of commissioning |     | are | not included |     | in this | analysis. The |                  |               |             |     |          |     |        |
| --------------------- | --- | --- | ------------ | --- | ------- | ------------- | ---------------- | ------------- | ----------- | --- | -------- | --- | ------ |
|                       |     |     |              |     |         |               | in this analysis | are therefore | nonetheless |     | alarming | and | under- |
averagefailureratesandthecorrespondingconfidenceintervals
|     |     |     |     |     |     |     | line the urgent | need to understand |     | and | eliminate | the | causes of |
| --- | --- | --- | --- | --- | --- | --- | --------------- | ------------------ | --- | --- | --------- | --- | --------- |
calculatedforeachofthesethreeturbinegenerationsareshown
converterfailure.
inFig.9forbothDFIG-basedandEESG-basedsystems.Note
| that only | failures of | the phase | module | as  | the core | component |     |     |     |     |     |     |     |
| --------- | ----------- | --------- | ------ | --- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- |
F. SeasonalVariationofPhase-ModuleFailureRates
| of the power | converter | are | considered | in  | this as | well as in the |     |     |     |     |     |     |     |
| ------------ | --------- | --- | ---------- | --- | ------- | -------------- | --- | --- | --- | --- | --- | --- | --- |
subsequentanalyses. Insearchofthemaincausesanddriversofconverterfailure,
The analysis is carried out in two steps: In the first step, all temporal or spatial patterns in the failure behavior can often
phase-modulefailuredatafromtherespectivegroupofturbines provide valuable indications. In order to identify a potential
isincluded(seethedarker-coloredbarsinFig.9).Inthisway, season-dependent clustering of failures, the average phase-
theanalysiscanbebasedonthelargestpossibleamountofdata. modulefailureratesofWTfleetsindifferentregionsandclimate
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

FISCHERetal.:RELIABILITYOFPOWERCONVERTERSINWINDTURBINES:EXPLORATORYANALYSISOFFAILUREANDOPERATINGDATA 6339
Fig.9. Phase-modulefailureratesofDFIG-basedandEESG-basedwindturbinescommissioneduntil2005,during2006–2010,andduring2011–2015.
Fig.10. Seasonalvariationofphase-modulefailureratesinIndia(left,basedondatafrom590WToperatingyears)andScandinavia(right,933WToperating
years)withcorrespondingmonthlyaveragevaluesofwindspeed,ambienttemperatureandrelativehumidity.
zonesareevaluatedforeachmonthoftheyear.Allavailablefail- a maximum in August. A less extreme, but still clear sea-
uredatafromthespecificregionandmonthareincludedinthis sonal clustering of failure is found in Scandinavia: According
analysis, so that, e.g., the average failure rate obtained for the to the corresponding top right diagram in Fig. 10, the high-
monthofJanuaryisbasedonfailuresinJanuary2003,January estphase-module failureratesareobserved duringthemonths
2004, ...,andJanuary2017. July to October. It is interesting to note that in case of the
Aparticularlyclearseasonalclusteringoffailuresisobserved Scandinavian fleet, a clustering of phase-module failures in
inthefleetinIndia,cf.thetopleftdiagraminFig.10.Here,the thisseasoncanbeobservedindependentlyforWTofdifferent
phase-modulefailureratesduringthemonthsJune–September designs and manufacturers. In other regions such as Germany,
are considerably higher than during the rest of the year, with no similarly clear seasonal failure pattern is found. An inter-
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore. Restrictions apply.

| 6340 |     |     |     |     |     |     | IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019 |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
esting difference in this context is that the fleet in Germany is this purpose, the average active power in the 10 min interval
stronglydominatedbyWTwithair-cooledconverterswhilethe preceding failure is determined, as described in Section II-C.
investigated Indian and Scandinavian fleets use liquid-cooled For four groups of WT (with the group name indicating the
converters. generator concept and each group corresponding to a certain
Theseasonalvariationofthephase-modulefailurerateindi- manufacturer), the distribution of phase-module failure events
catesthatthelocalwindand/orclimateconditionshaveacrucial isplottedagainstthe(normalized)activepower,asshowninthe
|     |     |     |     |     |     |     | upper diagrams |     | in Fig. | 11. It | can be | noted | that failures | do not |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------- | ------ | ------ | ----- | ------------- | ------ |
influenceonconverterreliability.IntheIndianfleet,thefailure
onlyoccurintherangeof>0...100%oftheWTratedpower.
| cluster coincides | with | the period | of  | Monsoon, | which | is char- |     |     |     |     |     |     |     |     |
| ----------------- | ---- | ---------- | --- | -------- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Theyarealsoobservedinstates,inwhichnopowerisfedinto
| acterized | by strong winds | and | heavy | rains. | In order | to narrow |     |     |     |     |     |     |     |     |
| --------- | --------------- | --- | ----- | ------ | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
downthepotentialinfluencingfactors,themonthlyaverageval- the grid or in which the 10 min-average of the active power
uesofwindspeed,relativehumidity,ambienttemperature,and exceedstheratedpower.
Aparticularlyhighnumberofphase-modulefailuresisfound
dew-pointtemperaturehavebeenidentifiedforboththesitesin
IndiaandScandinavia.Incaseofthewindspeed,thesearede- infull-loadoperationaswellasatlowpart-loadoperationand
atP(cid:4)0kW.However,thesoleevaluationoftheabsolutenum-
termineddirectlyfromtheSCADAdataofWTintherespective
|         |             |        |             |     |             |     | ber of failures |     | (or their | distribution) |     | over the | active | power is |
| ------- | ----------- | ------ | ----------- | --- | ----------- | --- | --------------- | --- | --------- | ------------- | --- | -------- | ------ | -------- |
| region. | The monthly | values | of humidity | and | temperature | are |                 |     |           |               |     |          |        |          |
derivedfromclimatedataby[39],[40](Scandinavia,datafrom only of limited informative value because the operating time
|     |     |     |     |     |     |     | of a WT | in each | of the | ranges | or bins | of active | power | differs |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ------ | ------ | ------- | --------- | ----- | ------- |
[34])and[41](India)bymeansofaveragingovertheregionsof
interestaswellasoverseveralyears.Fig.10showsthemonthly considerably(cf.thediagramsinthemiddleofeachsubfigure
averagevaluesofthesequantitiesthroughthecourseoftheyear of Fig. 11). This is closely related to the differing frequency
|            |                 |            |     |      |                  |     | distributions | of  | wind | speed at | the different |     | sites. | In order to |
| ---------- | --------------- | ---------- | --- | ---- | ---------------- | --- | ------------- | --- | ---- | -------- | ------------- | --- | ------ | ----------- |
| and allows | a direct visual | comparison |     | with | the phase-module |     |               |     |      |          |               |     |        |             |
failurerates. obtainavisualrepresentationthatallowsconclusionsaboutthe
|     |     |     |     |     |     |     | risk of converter |     | failure | in different |     | operating | points, | for each |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------- | ------------ | --- | --------- | ------- | -------- |
IncaseoftheIndianWTfleet,thehighestaveragewindspeed
occursinJune.However,particularlyhighfailureratesarefound bintheabsolutenumberoffailuresisdividedbythetotaloper-
atingtimeinthisloadrange.Thisprocedureresultsinafailure
inAugust,whentheaveragewindspeedhasalreadydecreased
|               |            |       |           |             |      |         | rate or failure | frequency |     | for each | load | bin, | which | is shown in |
| ------------- | ---------- | ----- | --------- | ----------- | ---- | ------- | --------------- | --------- | --- | -------- | ---- | ---- | ----- | ----------- |
| considerably. | Similarly, | there | is little | correlation | with | the am- |                 |           |     |          |      |      |       |             |
normalizedforminthediagramsatthebottomofthesubfigures
bienttemperature.Incontrast,theexcessivephase-modulefail-
inFig.11.
uresfallintothemonthswithparticularlyhighrelativehumidity
(RH)anddew-pointtemperatures.IncaseoftheScandinavian If phase-module failure occurred fully independently of the
WToperatingpoint,thefailurerateinallbinsofactivepower
fleet,thereisaparticularlystrongcorrelationbetweenfailures
shouldbeapproximatelyequal.However,theresultsinthelower
andhighvaluesofthedew-pointtemperature.Here,inviewof
diagramsinFig.11showclearlythatthisisnotthecase.There
| themuchsmallervariationofRHinScandinavia |     |     |     |     | compared |     | to        |          |          |     |        |            |        |       |
| ---------------------------------------- | --- | --- | --- | --- | -------- | --- | --------- | -------- | -------- | --- | ------ | ---------- | ------ | ----- |
|                                          |     |     |     |     |          |     | is rather | a common | tendency |     | in all | considered | groups | of WT |
India,thedewpointispredominantlydeterminedbytheambient
temperature(andwiththatthewatervaporabsorptioncapacity that the failure rate increases with the active power that the
|              |             |          |            |     |              |     | WT feed    | into   | the grid. | (Note | that | in case | of turbines | with        |
| ------------ | ----------- | -------- | ---------- | --- | ------------ | --- | ---------- | ------ | --------- | ----- | ---- | ------- | ----------- | ----------- |
| of the air). | If there is | a mostly | unhindered |     | air exchange | be- |            |        |           |       |      |         |             |             |
|              |             |          |            |     |              |     | DFIG, this | is not | identical | with  | the  | active  | power       | Pconv pass- |
tweentheambientairandtheinterioroftheWTandelectrical
|           |                  |             |     |     |             |         | ing the      | power | converter; | see, | e.g., [28] | for       | a presentation | of        |
| --------- | ---------------- | ----------- | --- | --- | ----------- | ------- | ------------ | ----- | ---------- | ---- | ---------- | --------- | -------------- | --------- |
| cabinets, | a high dew-point | temperature |     | in  | the ambient | air re- |              |       |            |      |            |           |                |           |
|           |                  |             |     |     |             |         | the relation | of    | Pconv and  | PWT  | in a       | DFIG-WT.) | The            | full-load |
sultsinasimilarlyhighdewpointinsidetheconvertercabinet.
|         |                  |     |             |       |        |           | operation | is obviously |     | afflicted  | with | a higher | risk      | of failure |
| ------- | ---------------- | --- | ----------- | ----- | ------ | --------- | --------- | ------------ | --- | ---------- | ---- | -------- | --------- | ---------- |
| Because | of the fact that | the | temperature | level | of the | converter |           |              |     |            |      |          |           |            |
|         |                  |     |             |       |        |           | for the   | phase-module |     | components | than | the      | operation | at low     |
componentsismainlydeterminedbytheoperatingpointandthe
|          |                |         |          |                 |     |        | part load. | On closer | inspection |     | of Fig. | 11, | however, | there are |
| -------- | -------------- | ------- | -------- | --------------- | --- | ------ | ---------- | --------- | ---------- | --- | ------- | --- | -------- | --------- |
| settings | of the cooling | system, | the risk | of condensation |     | in the |            |           |            |     |         |     |          |           |
power converter increases directly with the dew-point temper- also major differences between the considered groups of WT:
|                                                |     |     |     |     |     |         | In the groups |      | “DFIG2” | and   | “IG+FPC,”     | the | failure | rates are |
| ---------------------------------------------- | --- | --- | --- | --- | --- | ------- | ------------- | ---- | ------- | ----- | ------------- | --- | ------- | --------- |
| ature.Therefore,theresultsshowninFig.10provide |     |     |     |     |     | astrong |               |      |         |       |               |     |         |           |
|                                                |     |     |     |     |     |         | particularly  | high | in the  | range | of >90...100% |     | of      | the rated |
indicationthathumidityandcondensationhavearelevantinflu-
|     |     |     |     |     |     |     | power. In | contrast, | the | group | “EESG” | stands | out | with an ex- |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --- | ----- | ------ | ------ | --- | ----------- |
enceontheemergenceofphase-modulefailures.Thisisfurther
|     |     |     |     |     |     |     | ceptionally | high | failure | rate | during | operation | above | the rated |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ------- | ---- | ------ | --------- | ----- | --------- |
supportedbythefrequentoccurrenceofconverterfailuresdur-
| ingtherestartofWTafterlongerperiodsofdowntime,whichis |     |     |     |     |     |     | power.  |        |         |     |          |        |         |         |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------- | ------ | ------- | --- | -------- | ------ | ------- | ------- |
|                                                       |     |     |     |     |     |     | A final | remark | is that | the | analysis | method | applied | in this |
widelyreportedfromthefieldandwhichislikelytobecaused
|                 |          |     |         |                 |     |           | section    | can provide | useful  | results | only   | if the  | number  | of fail-  |
| --------------- | -------- | --- | ------- | --------------- | --- | --------- | ---------- | ----------- | ------- | ------- | ------ | ------- | ------- | --------- |
| by condensation | (related | to  | missing | or insufficient |     | converter |            |             |         |         |        |         |         |           |
|                 |          |     |         |                 |     |           | ure events | is high     | enough. | The     | reason | of this | is that | the lower |
pre-heatingroutines)[1],[34].
|     |     |     |     |     |     |     | the number | of            | failures,    | the higher | is  | the uncertainty |     | and vari-   |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------------- | ------------ | ---------- | --- | --------------- | --- | ----------- |
|     |     |     |     |     |     |     | ation of   | the resulting | failure-rate |            | bar | diagrams,       | so  | that a sub- |
G. InfluenceoftheWTOperatingPoint
|     |     |     |     |     |     |     | stantiated | recognition   |     | of particularly |          | failure-critical |     | operating   |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | --------------- | -------- | ---------------- | --- | ----------- |
|     |     |     |     |     |     |     | points is  | not possible. |     | The             | diagrams | shown            | in  | Fig. 11 are |
Anotherquestionthatcanbeinvestigatedbymeansofacom-
bined analysis of failure and operating data is, from which based on numbers of 45–245 phase-module failures per WT
group.
| operating | point the phase-module |     | failures |     | have occurred. | For |     |     |     |     |     |     |     |     |
| --------- | ---------------------- | --- | -------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

FISCHERetal.:RELIABILITYOFPOWERCONVERTERSINWINDTURBINES:EXPLORATORYANALYSISOFFAILUREANDOPERATINGDATA 6341
Fig.11. Analysisoftheoperatingpointprecedingphase-modulefailuresforwindturbinesoffourdifferentmanufacturers(designationindicatesthegenerator
type):distributionoffailureeventsoverbinsofactivepower(top),operatingtime(middle),andresultingfailurerate(bottom)ineachactive-powerbin.
IV. CONCLUSION This study has investigated failures of the complete main
|        |               |     |              |                 |                |     | power-converter |          | system, | which is | defined   | to include—besides |        |      |
| ------ | ------------- | --- | ------------ | --------------- | -------------- | --- | --------------- | -------- | ------- | -------- | --------- | ------------------ | ------ | ---- |
| With   | the objective | to  | contribute   | to the          | identification |     | of root         |          |         |          |           |                    |        |      |
|        |               |     |              |                 |                |     | the power       | modules, | their   | driver   | boards as | well as            | the DC | link |
| causes | and drivers   | of  | the frequent | power-converter |                |     | failures        |          |         |          |           |                    |        |      |
capacitorsandbusbarsascorecomponents—theconvertercool-
| in wind | turbines, | this | paper | has presented | the | results | of an       |           |          |     |          |          |     |      |
| ------- | --------- | ---- | ----- | ------------- | --- | ------- | ----------- | --------- | -------- | --- | -------- | -------- | --- | ---- |
|         |           |      |       |               |     |         | ing system, | converter | control, | and | filters, | but also | the | main |
in-depthanalysisoffailureandoperatingdata.Theunderlying
∼7400 circuitbreakerandthegrid-couplingcontactor.Withinthecon-
| data basis | covers | in total |     | operating |     | years of | 2734 |     |     |     |     |     |     |     |
| ---------- | ------ | -------- | --- | --------- | --- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
vertersystem,theabovementionedcorecomponents,whichare
| variable-speed |             | wind turbines |                   | of a multitude     | of           | manufacturers |                   |                    |          |             |              |            |              |        |
| -------------- | ----------- | ------------- | ----------------- | ------------------ | ------------ | ------------- | ----------------- | ------------------ | -------- | ----------- | ------------ | ---------- | ------------ | ------ |
|                |             |               |                   |                    |              |               | denoted           | “phase-module”     |          | components, | stand        | out        | with         | the on |
| and types,     | operating   | at            | onshore           | and                | offshore     | sites on      | four              |                    |          |             |              |            |              |        |
|                |             |               |                   |                    |              |               | average           | highest            | portion  | of failures | (neglecting  |            | the category |        |
| different      | continents, | which         |                   | is to the          | authors’     | knowledge     |                   |                    |          |             |              |            |              |        |
|                |             |               |                   |                    |              |               | “Other”           | with miscellaneous |          | minor       | items).      | Due to     | the fact     | that   |
| the to date    | most        | comprehensive |                   | converter-specific |              | field-data    |                   |                    |          |             |              |            |              |        |
|                |             |               |                   |                    |              |               | this component    |                    | category | dominates   | also         | the annual | average      |        |
| collection     | in the      | world.        | Methodologically, |                    | this         | paper         | takes a           |                    |          |             |              |            |              |        |
|                |             |               |                   |                    |              |               | converter-related |                    | repair   | costs and   | downtimes,   | as         | a previous   |        |
| step beyond    | previous    | failure-rate  |                   | based              | analyses,    | on the        | one               |                    |          |             |              |            |              |        |
|                |             |               |                   |                    |              |               | study of          | the authors        | had      | shown,      | the majority | of         | analyses     | in     |
| hand by        | quantifying | the           | uncertainty       | of                 | the provided |               | failure           |                    |          |             |              |            |              |        |
thispaperhavebeenfocusedonthephase-modulecategory.
| rates by | means | of confidence |     | intervals, | on the | other hand | by                |     |     |                                   |     |     |     |     |
| -------- | ----- | ------------- | --- | ---------- | ------ | ---------- | ----------------- | --- | --- | --------------------------------- | --- | --- | --- | --- |
|          |       |               |     |            |        |            | Onaverage,0.48a−1 |     |     | converter-systemfailuresperWThave |     |     |     |     |
combiningfailureandoperatingdatatoobtaindeeperinsights
beenobservedintheinvestigatedturbinefleet.Amongtheseare
intheemergenceofconverterfailure.
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

| 6342 |     |     |     |     |     |     |     | IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019 |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
a−1
0.16 phase-module failures per turbine. The analysis has Subsequentworkseekstoutilizethefielddatatoinvestigate
revealedlargedifferencesbetweengroupsofWTwithdifferent theeffectofdesignfactorsandoperatinghistoriesonconverter
generator-converter concepts: Among the compared concepts, reliability,tolinkthedata-analysisresultswiththefindingsof
thegroupofEESG-basedWTstandsoutwiththehighestcon- comprehensivepost-mortemanalysesofdamagedcomponents,
verterandphase-modulereliability.Itisfollowedbythegroup andtoderiveonthisbasisrecommendedmeasuresforenhancing
ofWTwithDFIG,whiletheconsideredWTwithIG+FPCshow thepower-converterreliabilityinwindturbines.
thelowestphase-modulereliability.Particularlyinterestingre-
sultshavebeenobtainedforDFIG-basedWT,whichaccording ACKNOWLEDGMENT
totheliteraturearewidelyconsideredtohaveaparticularlyhigh
|     |     |     |     |     |     |     |     | This work | was | carried |     | out within |     | the “Fraunhofer- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------- | --- | ---------- | --- | ---------------- | --- |
riskofconverterfailureduetotheseverethermalcyclingofthe
|     |     |     |     |     |     |     |     | Innovationscluster |     | Leistungselektronik |     |     |     | fu¨r regenerative |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ------------------- | --- | --- | --- | ----------------- | --- |
powersemiconductorsintheirmachine-sidepart:Oneisthatthe
|            |            |     |         |       |        |          |          | Energieversorgung”. |     | The           | authors | would             | like | to thank | their      |
| ---------- | ---------- | --- | ------- | ----- | ------ | -------- | -------- | ------------------- | --- | ------------- | ------- | ----------------- | ---- | -------- | ---------- |
| converters | in DFIG-WT |     | are not | found | in the | last row | with re- |                     |     |               |         |                   |      |          |            |
|            |            |     |         |       |        |          |          | project partners    |     | for providing |         | the comprehensive |      |          | field data |
specttoreliabilityandthatinfactverylowphase-modulefailure
underlyingthiswork.
rateshavebeenfoundinDFIGturbinesofcertainmanufactur-
ers.Theotherinterestingresultinthiscontextisthatthereare
REFERENCES
| not generally | more | IGBT-module |     | failures | in  | the machine-side |     |     |     |     |     |     |     |     |     |
| ------------- | ---- | ----------- | --- | -------- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thaninthegrid-sideconverterofDFIG-turbines.Thesefindings
[1] A.Bartschatetal.,“Zuverla¨ssigeLeistungselektronikfu¨rWindenergiean-
questionthehighfailureriskofDFIG-converterspostulatedin lagen (Reliable Power Electronics for Wind Turbines),” Final project
the literature and with that also the underlying widespread as- report of the Fraunhofer-Innovationscluster Leistungselektronik fu¨r re-
generativeEnergieversorgung,FraunhoferIWES,ed.Stuttgart,Germany:
sumptionthataccumulatingthermal-cyclinginducedfatigueis
Fraunhofer-Verlag,2018.
the lifetime-determining mechanism in WT power converters. [2] Fraunhofer IWES, “Website of the Fraunhofer innovation cluster on
[Online].
The observation that power converters in WT exhibit early- power electronics for renewable energy supply.” Available:
www.power4re.de/en.html.Accessed:2018.
failure characteristics [8], [11], [38] instead of a degradation- [3] V.YaramasuandB.Wu,ModelPredictiveControlofWindEnergyCon-
dominated failure behavior and the fact that no signs of fa- versionSystems.Hoboken,NJ,USA:Wiley,Feb.2017.
tigue damage could be found in field-returned power modules [4] K.Ma,PowerElectronicsfortheNextGenerationWindTurbineSystem.
NewYork,NY,USA:Springer,2015.
ofWTconverters[1],[35]providefurtherimportantindications [5] J.BuenoGayo,“FinalpublishablesummaryofresultsofprojectReli-
againsttherelevanceoffatigue-basedconverterfailuresinthis awind,”Gamesa,Madrid,Spain,2011.
[6] P.MatthewsandT.Ahmad,“Improvingenergyyieldthroughco-ordinated
application.
windfarmcontrol,”inPresentationattheWindFarmDataManagement
Anotherimportantfindingistheresultthattheconvertersand
andAnalysisForum,Hamburg,Germany,Oct.2016.
in particular the phase modules in WT of certain manufactur- [7] F.SpinatoandP.Tavner,“Reliabilityofwindturbinesubassemblies,”IET
|             |              |        |     |     |          | a−1 |              | RenewablePowerGener.,vol.3,no.4,pp.387–401,2009. |     |     |     |     |     |     |     |
| ----------- | ------------ | ------ | --- | --- | -------- | --- | ------------ | ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
| ers achieve | failure-rate | levels | as  | low | as 0.034 |     | per turbine, |                                                  |     |     |     |     |     |     |     |
[8] F.Spinato,TheReliabilityofWindTurbines,Ph.D.dissertation,Energy
whichprovesthatarelativelyhighreliabilityisinfactachievable
GroupoftheSchoolofEng.,DurhamUniv.,Durham,U.K.,2008.
withtheexistingmodular2-levelIGBT-basedlow-voltagecon- [9] S. Faulstich, B. Hahn, and P. Tavner, “Wind turbine downtime and its
importanceforoffshoredeployment,”WindEnergy,vol.14,no.3,pp.327–
| verter technology. |     | Research | should | therefore |     | prioritize | under- |     |     |     |     |     |     |     |     |
| ------------------ | --- | -------- | ------ | --------- | --- | ---------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
337,2010.
standingandlearningfromthesedifferencesinfield-reliability
[10] S.Faulstich,P.Lyding,andB.Hahn,“Electricalsubassembliesofwind
levels before aiming at further chip-technology enhancements turbines–Asubstantialriskfortheavailability,”inProc.Eur.WindEnergy
suchas,e.g.,newmaterialsforenhancedpower-cyclingdesign Conf.,Exhib.EWEC2010,Warsaw,Poland,2010,pp.129–133.
[11] J.Carroll,A.McDonald,O.B.Martin,D.McMillan,andR.Bakhshi,“Off-
life.
shorewindturbinesub-assemblyfailureratesthroughtime,”inProc.Eur.
AcomparisonofthefailureratesofWTcommissionedfrom WindEnergyAssoc.Annu.Conf.Exhib.,Paris,France,2015,pp.1–5.
1997 to 2015 has shown that the problem of high converter [12] J.Carroll,A.McDonald,andD.McMillan,“Failurerate,repairtimeand
unscheduledO&Mcostanalysisofoffshorewindturbines,”WindEnergy,
failure rates in not limited to old fleets, but that the average vol.19,no.6,pp.1107–1119,2015.
phase-modulereliabilityhasnotincreasedovertheWTgener- [13] J.Carroll,A.McDonald,andD.McMillan,“Reliabilitycomparisonof
ations.Thisstressesthecontinuednecessitytounderstandand wind turbines with DFIG and PMG drive trains,” IEEE Trans. Energy
Convers.,vol.30,no.2,pp.663–670,Jun.2015.
| eliminate | the causes | of  | converter | failure | also | in contemporary |     |              |        |         |        |            |          |         |          |
| --------- | ---------- | --- | --------- | ------- | ---- | --------------- | --- | ------------ | ------ | ------- | ------ | ---------- | -------- | ------- | -------- |
|           |            |     |           |         |      |                 |     | [14] Y. Lin, | L. Tu, | H. Liu, | and W. | Li, “Fault | analysis | of wind | turbines |
windturbines. in China,” Renewable Sustain. Energy Rev., vol. 55, pp. 482–490,
2016.
| In search | of factors | promoting |     | failure, | the | observation | of  | a   |     |     |     |     |     |     |     |
| --------- | ---------- | --------- | --- | -------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[15] C.Buscaetal.,“Anoverviewofthereliabilitypredictionrelatedaspects
| strong seasonal | variation |     | of phase-module |     | failure |     | rates in WT |     |     |     |     |     |     |     |     |
| --------------- | --------- | --- | --------------- | --- | ------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
ofhighpowerIGBTsinwindpowerapplications,”Microelectron.Rel.,
fleets in India and Scandinavian has revealed valuable indica- vol.51,pp.1903–1907,2011.
|                                                        |             |          |            |          |      |          |           | [16] E. E.     | Kostandyan    | and      | K. Ma,         | “Reliability | estimation |           | with uncer- |
| ------------------------------------------------------ | ----------- | -------- | ---------- | -------- | ---- | -------- | --------- | -------------- | ------------- | -------- | -------------- | ------------ | ---------- | --------- | ----------- |
| tions: The                                             | finding     | that the | months     | with     | high | failure  | rates co- |                |               |          |                |              |            |           |             |
|                                                        |             |          |            |          |      |          |           | tainties       | consideration |          | for high       | power IGBTs  |            | in 2.3 MW | wind tur-   |
| incide with                                            | the periods |          | of highest | absolute |      | humidity | suggests  |                |               |          |                |              |            |           |             |
|                                                        |             |          |            |          |      |          |           | bine converter |               | system,” | Microelectron. |              | Rel., vol. | 52, pp.   | 2403–2408,  |
| thathumidityand/orcondensationplayanimportantroleinthe |             |          |            |          |      |          |           | 2012.          |               |          |                |              |            |           |             |
[17] M.MusallamandC.M.Johnson,“Impactofdifferentcontrolschemeson
| emergence | of phase-module |     | failures. |     | Finally, | an investigation |     |     |     |     |     |     |     |     |     |
| --------- | --------------- | --- | --------- | --- | -------- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thelifeconsumptionofpowerelectronicmodulesforvariablespeedwind
| of the WT | operating | points | preceding |     | phase-module |     | failures |     |     |     |     |     |     |     |     |
| --------- | --------- | ------ | --------- | --- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
turbines,”inProc.14thEur.Conf.PowerElectron.Appl.,Birmingham,
hasshownthat,inspiteofcertainWT-typespecificdifferences, U.K.,2011,pp.1–9.
theriskoffailureishighestduringoperationcloseto,ator(as [18] M.Ikonen,“PowercyclinglifetimeestimationofIGBTpowermodules
|     |     |     |     |     |     |     |     | based | on chip | temperature | modeling,” | Ph.D. | dissertation, |     | Dept. Elect. |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------- | ----------- | ---------- | ----- | ------------- | --- | ------------ |
observedinsomeWTtypes)evenaboveratedpower,i.e.,under Eng.,Inst.EnergyTechnol.,LappeenrantaUniv.Technol.,Lappeenranta,
| theimpactofhighelectricalloading. |     |     |     |     |     |     |     | Finland,2012. |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore.  Restrictions apply.

FISCHERetal.:RELIABILITYOFPOWERCONVERTERSINWINDTURBINES:EXPLORATORYANALYSISOFFAILUREANDOPERATINGDATA 6343
[19] K.Ma,M.Liserre,F.Blaabjerg,andT.Kerekes,“Thermalloadingand [40] SwedishMeteorologicalandHydrologicalInstitut(SMHI),Norrko¨ping,
lifetimeestimationforpowerdeviceconsideringmissionprofilesinwind Sweden,2011.
powerconverter,”IEEETrans.PowerElectron.,vol.30,no.2,pp.590– [41] “Weatherbase,” 2017. [Online]. Available: www.weatherbase.com. Ac-
602,Feb.2015. cessedon:Jun.1,2017.
[20] M. Bohlla¨nder, “Lastwechseltestbasierte Lebensdaueranalysemethoden
fu¨rLeistungshalbleiterinOffshore-Windenergieanlagen(Power-cycling
based lifetime analysis methods for power semiconductors in offshore Katharina Fischer received the Dipl.-Ing. degree
windturbines),”Ph.D.dissertation,Dept.PowerElectron.Electromag- in electrical engineering and the Ph.D. degree in
neticCompatibility,ChemnitzUniv.Technol.,Chemnitz,Germany,2014. mechanical engineering from Leibniz University
[21] M.Held,P.Jacob,G.Nicoletti,P.Scacco,andM.H.Poech,“Fastpower Hannover,Germany,in2002and2008,respectively.
cyclingtestofIGBTmodulesintractionapplication,”inProc.Int.Conf. She was a Postdoctoral Researcher with the
PowerElectron.DriveSyst.,1997,pp.425–430. Chalmers University of Technology, Gothenburg,
[22] R.Bayerer,T.Hermann,T.Licht,J.Lutz,andM.Feller,“Modelforpower Sweden, until she joined Fraunhofer IWES in
cyclinglifetimeofIGBTmodules-Variousfactorsinfluencinglifetime,” HanoverasaSeniorScientistin2012.Herresearch
in Proc. 5th Int. Conf. Integr. Power Syst., Nu¨rnberg, Germany, 2008, interestsincludethereliability,conditionmonitoring,
pp.1–6. andmaintenanceofwindturbines,withaparticular
[23] U.ScheuermannandR.Schmidt,“Newlifetimemodelforadvancedpower focusontheirpower-electronicconverters.
moduleswithsinteredchipsandoptimizedalwirebonds,”inProc.Power
Convers.Intell.MotionEur.,Nuremberg,Germany,2013,pp.810–817.
[24] M. Bartram, I. V. Bloh, and R. D. Doncker, “Doubly-fed machines in
KarolinePelkareceivedtheM.Sc.degreeinmathe-
wind-turbinesystems:IsthisapplicationlimitingthelifetimeofIGBT
maticsfromtheUniversityofMu¨nster,Germany,in
frequency-converters?,”inProc.35thAnnu.IEEEPowerElectron.Spec.
2015.
Conf.,Aachen,Germany,2004.
Since2016,shehasbeenworkingasaResearch
[25] M.Bartram,IGBT-UmrichtersystemefuerWindkraftanlagen:Analyseder
AssociateintheGroupTechnicalReliabilityatFraun-
Zyklenbelastung,Modellbildung,OptimierungundLebensdauervorher-
hofer IWES in Hanover. Her research interests in-
sage (IGBT converter systems for wind turbines: Analysis of cycling
clude statistical methods and models for reliability
stress, modeling, optimization and life prediction), Ph.D. dissertation,
analysisofwindturbines.
Inst.PowerElectron.Elect.Drives,RWTHAachen,Germany,2005.
[26] H.Wangetal.,“Transitioningtophysics-of-failureasareliabilitydriver
inpowerelectronics,”IEEEJ.Emerg.Sel.TopicsPowerElectron.,vol.2,
no.1,pp.97–114,Mar.Mar.2014.
[27] L.Wei,R.Kerkman,R.Lukaszewski,H.Lu,andZ.Yuan,“Analysisof
IGBTpowercyclingcapabilitiesusedindoublyfedinductiongenerator
ArneBartschatreceivedtheM.Sc.degreeinrenew-
windpowersystem,”IEEETrans.Ind.Appl.,vol.47,no.4,pp.1794–1801,
ableenergysystemsfromtheHTWBerlin,Germany,
Jul./Aug.2011.
in2014.
[28] F.FuchsandA.Mertens,“Steadystatelifetimeestimationofthepower
From2014to2017,heworkedasaResearchAs-
semiconductorsintherotorsideconverterofa2MWDFIGwindturbine
sociateatFraunhoferIWESwithafocusonmodeling
viapowercyclingcapabilityanalysis,”inProc.Eur.Conf.PowerElectron.
andsimulationofwindturbinesandreliabilitystudies
Appl.,Birmingham,U.K.,2011,pp.1–8.
ofmechanicalandelectricaldrive-traincomponents.
[29] R.Pittini,S.D’Arco,M.Hernes,andA.Petterteig,“Thermalstressanal-
Since2018,hehasbeentheHeadoftheLargeBear-
ysisofIGBTmodulesinVSCsforPMSGinlargeoffshorewindenergy
ingsGroupofFraunhoferIWESinHamburg.
conversionsystems,”inProc.14thEur.Conf.PowerElectron.Appl.,2011,
pp.1–10.
[30] S.D’Arco,T.Undeland,M.Bohlla¨nder,andJ.Lutz,“Asimplifiedalgo-
rithmforpredictingpowercyclinglifetime indirectdrivewindpower
systems,” in Proc. 9th Int. Multi-Conf. Syst., Signals Devices, 2012,
pp.1–6. BerndTegtmeierreceivedtheDipl.-Ing.degreein
[31] C.J.Smith,C.J.Crabtree,andP.C.Matthews,“Impactofwindconditions electricalengineeringfromtheUniversityofApplied
onthermalloadingofPMSGwindturbinepowerconverters,”IETPower SciencesinLippe/Ho¨xter,Germany,in1997.
Electron.,vol.10,no.11,pp.1268–1278,Sep.2017. HeworkedasaDevelopmentEngineerindifferent
[32] Q.Zhou,S.Xue,J.Li,C.Xiang,andS.Chen,“Evaluationofwindpower positionsintheindustryandatDLRinthefieldof
converterreliabilityconsideringmulti-timescaleandmissionprofiles,”in militaryandaerospaceengineeringandsubsequently
Proc.IEEEInt.Conf.HighVoltageEng.Appl.,Chengdu,China,2016, forsevenyearsataGermanwind-turbinemanufac-
pp.1–4. turer.Since2014,hehasbeenaResearchAssociate
[33] S.Konrad,“EinBeitragzurAuslegungundIntegrationspannungsgespeis- at Fraunhofer IWES, with a main focus on power
terIGBT-Wechselrichter(Acontributiontothedesignandintegrationof converters,nacelleinstrumentationandtesting,and
IGBT-basedvoltagesourceconverters),”Ph.D.dissertation,Tech.Univ. electricalcertification.
Chemnitz,Ilmenau,Germany,1997.
[34] K.Fischer,T.Stalin,H.Ramberg,T.Thiringer,J.Wenske,andR.Karlsson,
“Investigationofconverterfailureinwindturbines,”Elforskreport12:58,
Stockholm,Sweden,2012. Diego Coronado received the double degree in
[35] K.Fischeretal.,“Field-experiencebasedroot-causeanalysisofpower- mechanical engineering from the Ecole Nationale
converterfailureinwindturbines,”IEEETrans.PowerElectron.,vol.30, d’Inge´nieursdeMetzinFranceandtheUniversidad
no.5,pp.2481–2492,May2015. Tecnolo´gicadePereirain2011.Hereceivedthemas-
[36] K.FischerandJ.Wenske,“Towardsreliablepowerconvertersforwind ter’sdegreeinrenewableenergyfromMinesParis-
turbines:Field-databasedidentificationofweakpointsandcostdrivers,” Tech,France,in2013.
inProc.EWEAConf.,Paris,France,2015,pp.117–120. He worked as a Research Associate for Fraun-
[37] L.J.BainandM.Engelhardt,Statisticalanalysisofreliabilityandlife hoferIWESfor5yearsdoingresearchinthefieldof
testingmodels:Theoryandmethods.NewYork,NY,USA:MarcelDekker, conditionmonitoringforwindturbines.Heworked
1991. asaServiceandDiagnosisEngineeratOELCHECK
[38] K.PelkaandK.Fischer,“Failurebehaviourofpowerconvertersinwind GmbH and he is currently a Mechanical and Op-
turbines,”inProc.13thGermanWindEnergyConf.DEWEK,Bremen, erationEngineeratCGN EuropeEnergyinParis, France, beingresponsible
Germany,2017. for the supervision of wind farms including condition monitoring and asset
[39] DanishMeteorologicalInstitute(DMI),Kopenhagen,Denmark,2011. management.
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore. Restrictions apply.

6344 IEEETRANSACTIONSONPOWERELECTRONICS,VOL.34,NO.7,JULY2019
Christian Broer received the Dipl.-Ing. degree in Jan Wenske received the Dipl.-Ing. degree in me-
mechanical engineering from Leibniz University chanical engineeringandthePh.D. degreeinelec-
Hannover,Germany,in2005. tricalengineeringfromtheTechnicalUniversityof
HeworkedasaResearchAssociateattheInsti- Clausthal,Germany,in1994and2000,respectively.
tuteofMaterialsScienceofthesameuniversity.For Forseveralyearsheworkedindifferentpositions
severalyears,heworkedindifferentpositionsinthe in the industry. Since 2011, he has been the Head
industrydealingwithR&Dofpowerelectronicsand of the Department of Wind-Turbine and System-
satelliteinstrumentation.In2012,hejoinedFraun- TechnologyandtheDeputyDirectorofFraunhofer
hoferIWESandleadstheDepartmentofReliability IWES in Bremerhaven. Since 2013, he has been a
andValidationinHanoverandHamburg,Germany, Professorforwind-turbinetechnologyatUniversity
since2014. ofBremen,Germany.
Authorized licensed use limited to: Technische Universitaet Muenchen. Downloaded on June 15,2026 at 08:55:08 UTC from IEEE Xplore. Restrictions apply.