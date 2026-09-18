21 January 2025
Ancillary Services in Germany: Present, Future and
the Role of Battery Storage Systems
1 1 2
Prashant Pant , Thomas Hamacher , Vedran S Perić
1. Technical University of Munich
2. Aarhus University
Abstract
The growing integration of renewable energy sources poses new challenges to the existing power
system infrastructure. These challenges include reducing power system inertia, resulting in faster
frequency dynamics and reduced stability margins. Additionally, the limited capacity of power
electronics-based inverter units reduces the sources of reactive power generation, jeopardizing
voltage stability. To address these issues, system operators are increasingly required to procure
ancillary services to ensure reliable power system operation. This paper presents a comprehensive
literature review of the current state of the ancillary services market in Germany, examining
the various procurement schemes currently employed and the economics of services. It also
highlights the insufficiency of existing ancillary services and identifies potential services that may
be necessary to stabilize Germany’s future power systems, particularly as renewable energy sources
continue to gain prominence. This review aims to benefit academics, researchers, practitioners, and
policymakers by enabling them to make informed decisions and effectively navigate the changing
energy landscape in Germany.
Keywords
ancillary services, battery energy storage, fast frequency response (ffr), frequency stability, grid
reinforcement, power, energy and industry applications, voltage stability
Posted on 21 January 2025 — CC-BY 4.0 — This is a preprint and has not been peer reviewed. Data may be preliminary. — https://
doi.org/10.36227/techrxiv.171177336.66870344/v2

Ancillary Services in Germany: Present, Future and
the Role of Battery Storage Systems
Prashant Pant Thomas Hamacher Vedran S. Peric´
Technical University of Technical University of Aarhus University
Munich, Germany Munich, Germany Denmark
prashant.pant@tum.de thomas.hamacher@tum.de vedran.peric@ece.au.dk
Abstract—The growing integration of renewable energy Energy storage systems, particularly battery energy storage
sourcesposesnewchallengestotheexistingpowersysteminfras- systems (BESS), have the potential to provide grid flexibility
tructure.Thesechallengesincludereducingpowersysteminertia,
andarangeofancillaryservicesacrossvarioustimeintervals,
resulting in faster frequency dynamics and reduced stability
spanning from milliseconds to hours [4], [5]. Their ability to
margins. Additionally, the limited capacity of power electronics-
based inverter units reduces the sources of reactive power quickly adjust output and low operating costs make BESS
generation,jeopardizingvoltagestability.Toaddresstheseissues, units well-suited for supplying frequency reserves, voltage
system operators are increasingly required to procure ancillary control,andenablingenergyarbitrage[5]–[7].Thedecreasing
services to ensure reliable power system operation. This paper
costs of battery technologies further enhance their potential
presentsacomprehensiveliteraturereviewofthecurrentstateof
in future power systems. Projections indicate that global cost
theancillaryservicesmarketinGermany,examiningthevarious
procurement schemes currently employed and the economics of reductions of 75% by 2050 are expected in the best case [6].
services. It also highlights the insufficiency of existing ancillary Furthermore,thecomingfleetofsecond-lifebatteriescanalso
services and identifies potential services that may be necessary be deployed for providing grid ancillary services [8].
to stabilize Germany’s future power systems, particularly as
renewable energy sources continue to gain prominence. This
reviewaimstobenefitacademics,researchers,practitioners,and
policymakers by enabling them to make informed decisions and
effectively navigate the changing energy landscape in Germany. Large scale storage
Index Terms—Ancillary services, battery energy storage, grid Industrial storage
reinforcement, frequency stability, voltage stability. Home storage
I. INTRODUCTION
The energy policy of the European Union is focused on
achieving climate neutrality by 2050. Germany, in particular,
aims to reach this milestone even earlier, targeting 2045
[1]. This commitment drives the ongoing transformation of
power systems to shift away from conventional power gen- Fig.1:BatterystoragestatusinGermanyasofOctober2023[9]
eration facilities and embrace decentralized renewable energy
Fig1 illustrates the current battery storage capacity in Ger-
sources, mainly focusing on wind and solar generation [2].
many, with home storage units contributing 15 GWh, while
Renewable energy sources now account for over 55% of total
large-scale storage (LSS) projects account for 2 GWh, and
electricity generation in Germany [3]. However, Germany’s
industrial storage adding up to 0.39 GWh. The projected
ambition extends further, aiming to cover at least 80% of
storage demand for LSS in Germany is expected to reach 104
gross electricity consumption with renewables by 2030. To
GWh by 2030, primarily to support grid stabilization efforts
achievethisgoal,thecountryhasimplementedtheRenewable
[10].Tomeetthisincreasingdemandandstimulateinvestment
EnergyAct(EEG)amendment,effectivefrom1January2023
in LSS projects, Germany has implemented a special tariff
[2]. This regulation promotes increased flexibility, mandating
scheme that provides a 20-year exemption from grid fees for
transmission system operators (TSOs) and distribution system
transmission-connected LSS projects commissioned between
operators (DSOs) to procure additional ancillary services to
2011 and 2026 [11]. Furthermore, on 17 October 2023, the
ensure the smooth functioning of the grid and maintain grid
German Energy Industry Act (EnWG) underwent significant
stability. The evolving energy landscape is anticipated to
amendments (§ 118, subsection 6), extending the grid fee
increase the demand for ancillary services and necessitate
exemption to include LSS installations commissioned prior
a greater diversity of such services, thereby reshaping the
to 2029 [12]. This particular amendment can further bolster
ancillary service markets.
investments in LSS projects.
Approximately two-thirds of the LSS capacity in Germany
The research work presented in this paper is supported by the Bavarian
ResearchFoundationwithinthescopeoftheresearchproject“KI-M-Bat." is dedicated to providing frequency ancillary services [10].

Table. I provides comprehensive information about various groupswereintroducedintotheenergymarket.Thebalancing
LSS projects in Germany and the services they offer. No- groups take care of the commercial flow of electricity, while
tably, the primary services observed in these projects are the physical power transfer is managed by the TSOs and
frequency support. However, each country’s demand for fre- DSOs. Every consumer or producer of electricity in Germany
quency control reserves remains nearly constant, leading to is signed to a balancing group [15]. The utility providers (or
market saturation and declining reserve capacity prices. This the retail suppliers) have an agreement with the balancing
situation poses a discouraging sign for new investments in group to supply electricity to its consumers. The balancing
the sector [13], [14]. As future grid operations require a group system takes electricity trading into account. At the
broader range of ancillary products to effectively manage the wholesale level, the traded electricity is transferred from the
intermittency associated with renewable energy generation, seller’s balancing group to the buyer’s balancing group.
| regulatory | bodies | shall | identify | these future | ancillary | services |     |     |     |     |     |     |     |     |     |
| ---------- | ------ | ----- | -------- | ------------ | --------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
and establish a profitable market structure that encourages  Energy Market Balancing Market Grid Secutity Measures
| BESS participation |              | in grid | stabilization. |     |     |     |     |                |     |                  |                    |     |     |                       |     |
| ------------------ | ------------ | ------- | -------------- | --- | --- | --- | --- | -------------- | --- | ---------------- | ------------------ | --- | --- | --------------------- | --- |
|                    |              |         |                |     |     |     |     | Power Exchange |     | Over the counter | Ancillary services |     |     | Federal network       |     |
|                    |              |         |                |     |     |     |     |                |     |          (OTC)   |                    |     |     | agency/ Grid operator |     |
| A. Paper           | contribution |         |                |     |     |     |     |                |     |                  |                    |     |     |                       |     |
FCR
Spot
Numerousstudieshavebeenconductedontherequirements Derivatives  Spot  contracts Control aFRR Redispatch
|     |     |     |     |     |     |     |     | market | market |     |     | reserves |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------ | --- | --- | -------- | --- | --- | --- |
for new ancillary services and potential market structures; Derivative Renewable
mFRR
however,nonehavespecificallyfocusedontheuniquerequire- contracts Reactive curtialment
|     |     |     |     |     |     |     |     | Futures | Day-ahead |     |     | power  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | --- | ------ | --- | --- | --- |
ments of the German power system. This research addresses Load shedding
|          |           |     |             |                |     |     |     |         | Intraday |     |     | Blackstart |     |     |     |
| -------- | --------- | --- | ----------- | -------------- | --- | --- | --- | ------- | -------- | --- | --- | ---------- | --- | --- | --- |
| this gap | by making | two | significant | contributions: |     |     |     | Options |          |     |     |            |     |     |     |
(continuous)
Grid reserve
| (i) Providing |     | a detailed | overview | of the | German | electricity |     |     | Intraday |     |     |     |     |     |     |
| ------------- | --- | ---------- | -------- | ------ | ------ | ----------- | --- | --- | -------- | --- | --- | --- | --- | --- | --- |
(auction)
| market, | the | status | quo of | the current | ancillary | services, |     |     |     |     |     |     |     |     |     |
| ------- | --- | ------ | ------ | ----------- | --------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Fig.2:SummaryofelectricitymarketdesigninGermany
| procurement    |     | mechanisms, |           | and the | economics | involved. |        |           |     |        |               |     |          |        |      |
| -------------- | --- | ----------- | --------- | ------- | --------- | --------- | ------ | --------- | --- | ------ | ------------- | --- | -------- | ------ | ---- |
| (ii) Examining |     | additional  | ancillary | service | products  |           | needed |           |     |        |               |     |          |        |      |
|                |     |             |           |         |           |           |        | According |     | to the | market report |     | from the | German | Fed- |
to accommodate the increasing share of renewable en- eral Cartel Office (Bundeskartellamt) [29], Germany had 890
| ergy           | sources | in the   | German | power          | system, | along      | with |              |        |       |               |             |             |     |         |
| -------------- | ------- | -------- | ------ | -------------- | ------- | ---------- | ---- | ------------ | ------ | ----- | ------------- | ----------- | ----------- | --- | ------- |
|                |         |          |        |                |         |            |      | DSOs in      | 2021.  | Their | core activity | is          | to maintain | and | monitor |
| implementation |         | examples |        | from different |         | countries. |      |              |        |       |               |             |             |     |         |
|                |         |          |        |                |         |            |      | distribution | grids. | In    | addition,     | they assign | consumption |     | points  |
Theremainderofthepaperisstructuredasfollows:Section
|     |     |     |     |     |     |     |     | to balancing |     | groups | and invoice | grid | usage | fees. | The TSOs |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------ | ----------- | ---- | ----- | ----- | -------- |
(II) provides an overview of the German electricity market, only invoice customers who are directly connected to the
brieflyexplainingthewholesaleandancillaryservicemarkets. transmission grid, including distribution grids. Through this
| Building | upon | that, Section | (III) | discusses | the | current | status |            |     |           |           |     |        |              |      |
| -------- | ---- | ------------- | ----- | --------- | --- | ------- | ------ | ---------- | --- | --------- | --------- | --- | ------ | ------------ | ---- |
|          |      |               |       |           |     |         |        | mechanism, | all | customers | connected |     | to the | distribution | grid |
of ancillary services in Germany, including both frequency pay a share of transmission costs. In addition to maintaining
and non-frequency services. The procurement mechanism and the transmission grid, it is the responsibility of TSOs to
prevailing prices of the ancillary services are presented, with maintain the grid frequency at a stable 50 Hz. To achieve
comments on the feasibility of BESS units for market partici- this,TSOsarerequiredtoprocurecapacityreservestobalance
pation.Section(IV)highlightstheinadequacyofcurrentancil-
|     |     |     |     |     |     |     |     | any load | and | generation | intermittency. |     | The balancing |     | groups |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | ---------- | -------------- | --- | ------------- | --- | ------ |
lary products in coping with the increasing share of inverter- offer the capacity reserves through auctions through various
based generation and recommends potential future ancillary market platforms. Fig. 2 summarizes the German electricity
| products | to maintain | grid | stability. | Section |     | (V) focuses | on  |     |     |     |     |     |     |     |     |
| -------- | ----------- | ---- | ---------- | ------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
marketandTable.IIpresentsthecurrentmarketshareofmajor
the potential frequency and non-frequency ancillary services German electricity companies at different operational levels.
| that may    | be required | in   | German   | power    | systems | in the           | near |              |     |             |        |     |     |     |     |
| ----------- | ----------- | ---- | -------- | -------- | ------- | ---------------- | ---- | ------------ | --- | ----------- | ------ | --- | --- | --- | --- |
|             |             |      |          |          |         |                  |      | A. Wholesale |     | electricity | market |     |     |     |     |
| future. The | section     | also | provides | examples | of      | similar services |      |              |     |             |        |     |     |     |     |
fromdifferentcountries.Section(VI)highlightsthecongestion Electricalenergycanbetradedateitherpowerexchangesor
problem in Germany and showcases the opportunities for grid over-the-counter(OTC),i.e.withbilateralcontracts.Thereare
reinforcement deferral through BESS at both the transmission different trading windows. The trading activities in Germany
anddistributionlevels.Finally,Section(VII)offersconcluding primarilytakeplaceonthreeexchanges:theEuropeanEnergy
remarksandrecommendationsregardingthefutureofancillary Exchange(EEX)locatedinLeipzig,theEPEXSPOTinParis,
services in Germany. andtheEnergyExchangeAustria(EXAA)inVienna[28].The
long-termcontractsupto7yearsinthefuturearetradedonthe
II. GERMANELECTRICITYMARKET:DEFINITIONS,LEGAL
|     |     |     |     |     |     |     |     | derivative | market | through | EEX | platform. | For | the intraday | and |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | ------- | --- | --------- | --- | ------------ | --- |
FRAMEWORK,ANDCURRENTSTATUS
day-aheadcontractpurchases,EPEXandEXAAplatformsare
| The EnWG |     | reform of | 1998 | incorporated | the | liberalization |     |           |               |     |         |            |      |           |     |
| -------- | --- | --------- | ---- | ------------ | --- | -------------- | --- | --------- | ------------- | --- | ------- | ---------- | ---- | --------- | --- |
|          |     |           |      |              |     |                |     | utilized, | respectively. |     | For the | OTC trade, | both | long-term | and |
of the electricity sector to decouple the physical flow (trans- short-term products exist.
missionanddistribution)andthecommercialflow(tradingand
sales) [15]. The liberalization of the energy sector introduced B. Ancillary services definition
various new players into the market. In 1999, the balancing The EU Directive 2019/944 [30]:

TABLEI:Listofselectedlarge-scalebatteryprojectsforprovidingdifferentgridservicesinGermany
| Project |     |     |     | Service |     |     | TechnicalDetails |     |     | Economic Details | (if avail- |
| ------- | --- | --- | --- | ------- | --- | --- | ---------------- | --- | --- | ---------------- | ---------- |
able)
EnergyStorageBordesholm[16] Ancillary services - FCR, Voltage sup- 10MWand15MWhcapacity,Saving C10 million investment,
port,Blackstart,Islanding of 12,000 tonnes CO2 equivalent per C1millionannualturnover
year,GridareaofTennet
BatteryStorageJardelund[17] Bulk energy services - Electric energy 48 MW and 50 MWh capacity, Con- C30millioninvestment
|     |     |     |     | timeshift(arbitrage),Reducewinden- |     |     | sists of                       | 10,000 Li-ion | battery modules, |     |     |
| --- | --- | --- | --- | ---------------------------------- | --- | --- | ------------------------------ | ------------- | ---------------- | --- | --- |
|     |     |     |     | ergycurtailment                    |     |     | OwnedbyMitsubishiandEnecoHold- |               |                  |     |     |
ing
EnergyStorageAlt-Daber[18] FCRparticipation,Balancingofvolatile 2 MW Storage and 2 MWh capacity, C1.3millioninvestment
|     |     |     |     | PVgeneration |     |     | Owner&Operator:UpsideGroup |     |     |     |     |
| --- | --- | --- | --- | ------------ | --- | --- | -------------------------- | --- | --- | --- | --- |
BigBatteryLausitz[19] Ancillary services - FCR, Electric en- 50 MW and 53 MWh capacity, 8,840 C25millioninvestment
|     |     |     |     | ergy time | shift (arbitrage), | Grid stabi- | batterymodules |     |     |     |     |
| --- | --- | --- | --- | --------- | ------------------ | ----------- | -------------- | --- | --- | --- | --- |
lization
SchwerinBatteryPark[20] Ancillary services - FCR, Black start, 10MWand15MWhcapacity,Funding C10millioninvestment
|     |     |     |     | Islanding,  | Electric energy | time shift | from           | the federal government | – R&D, |     |     |
| --- | --- | --- | --- | ----------- | --------------- | ---------- | -------------- | ---------------------- | ------ | --- | --- |
|     |     |     |     | (Arbitrage) |                 |            | privatefunding |                        |        |     |     |
Battery2ndLifeHamburg[21] Ancillaryservices-FCR 2 MW and 2.8 MWh capacity, 2600 N.A
modulesfrommorethan100cars,Sec-
ondlifebatteryproject,OwnedbyVat-
tenfall,supportedbyBMWandBosch
Pre-lifeHannover[22] Ancillaryservices-FCR 15 MW and 15 MWh capacity, 3000 N.A
|     |     |     |     |     |     |     | battery       | modules, Second | life bat-   |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --------------- | ----------- | --- | --- |
|     |     |     |     |     |     |     | tery project, | Owned by        | Daimler and |     |     |
StadtwerkeHannover
NetzboosterKupferzell[23] Gridcongestionmanagement/Immedi- 250 MW and 250 MWh capacity, C200millioninvestment
|     |     |     |     | ateupgradedeferral |     |     | OwnedbyTransnetBW,Tobefinished |     |     |     |     |
| --- | --- | --- | --- | ------------------ | --- | --- | ------------------------------ | --- | --- | --- | --- |
by2025
Netzbooster Audorf South and Otten- Gridcongestionmanagement/Immedi- 250 MW and 250 MWh capacity, C200millioninvestment
| hofen[7] |     |     |     | ateupgradedeferral |     |     | OwnedbyTennetGmbH,Tobefinished |     |     |     |     |
| -------- | --- | --- | --- | ------------------ | --- | --- | ------------------------------ | --- | --- | --- | --- |
by2034
M5BATAachen[24] Researchproject:Batterylifeoptimiza- 5MWand5MWhcapacity,Li-ionbat- C6.5 million grant from the
tion for Control reserve trading, Arbi- tery, lead acid and Na-NiCl2 batteries, German Federal Ministry for
|     |     |     |     | trage |     |     | Project | at E.ON Energy  | Centre, Sup- | EconomicAffairs |     |
| --- | --- | --- | --- | ----- | --- | --- | ------- | --------------- | ------------ | --------------- | --- |
|     |     |     |     |       |     |     | ported  | by: SMA, Exide, | Beta Motion  |                 |     |
andRWTHAachen
Wind+Storage project Brandenburg Gridflexibility,Avoidcurtailment Germany’sfirstWind+Storageproject, N.A
| Uckermark[25] |     |     |     |     |     |     | 3MWand3MWhcapacity |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- |
Six projects: Herne, Lünen, Bexbach, Ancillaryservices-FCR 90MWand45MWhcapacityintotal, C100millioninvestment
Fenne, Weiher, and Walsum Storage 15MWand7.5MWhcapacityeach,Li-
SystemNorthRhine-Westphalia[26] ionbatteries,OwnedbySTEAGGmbH
| Article. | 2 (48), |     | “ancillary | service" | “a  | service |     |     |     |     |     |
| -------- | ------- | --- | ---------- | -------- | --- | ------- | --- | --- | --- | --- | --- |
defines the as process ensuring transparency and non-discrimination. Addi-
necessary for the operation of a transmission or distribution tionally,TSOsconductanannualsystemanalysistodetermine
system, including balancing and non-frequency ancillary ser- the necessary ’reserve capacity’ for future grid stabilization
vices, but not including congestion management". through re-dispatch. The German Federal Network Agency
Article. 2 (49), defines the “non-frequency ancillary ser- (Bundesnetzagentur, BNetzA) reviews the system analysis
|          |            |      |      |              |                 |     | findings | and publishes | an annual | report on the requirement |     |
| -------- | ---------- | ---- | ---- | ------------ | --------------- | --- | -------- | ------------- | --------- | ------------------------- | --- |
| vice" as | “a service | used | by a | transmission | system operator |     |          |               |           |                           |     |
ordistributionsystemoperatorforsteadystatevoltagecontrol, for reserve generation capacity [33]. Control reserve and grid
fast reactive current injections, the inertia for local grid reserve collectively constitute ancillary services [34].
stability,short-circuitcurrent,blackstartcapabilityandisland
operation capability". C. Grid security measures from Grid Operators/ Federal
|       |         |          |      |           |           |      | Network | Agency |     |     |     |
| ----- | ------- | -------- | ---- | --------- | --------- | ---- | ------- | ------ | --- | --- | --- |
| § 13e | of EnWG | mandates | TSOs | to ensure | frequency | sta- |         |        |     |     |     |
bility within their respective control areas [31]. Similarly, § In an electrical power system, generation and demand must
12hofEnWG[32]mandatesbothTSOsandDSOstoprocure be balanced at every time, and it is the responsibility of TSO
non-frequencysystemservicesforvoltagestabilitywithintheir to ensure frequency stability, and TSOs divide their respon-
respective operation areas. To compensate for active power sibilities into smaller units known as balancing groups. Each
deficits, TSOs procure reserve capacity through the balancing balancinggroupensuresthatthegeneration,consumption,and
markets, a separate component from electricity markets. The traded energy within that specific zone are in equilibrium.
capacity reserve is established via a competitive tendering To facilitate this balance, each balancing group is required

TABLEII:Germanelectricitysectormarketsharein2023[27][28]
Sector Marketleaders Marketshare Totalnumber FRR
ofproviders
Transmission Amprion, 100%combined 4 manual RR
TransnetBW,
TenneT,50Hertz
Distribution EnBW,E.ON, Thebig4 Approx.890
RWE,Vattenfall, companiesown DSOs,with
andoperatea about700of
significant themownedby
portion(25.8%) Stadtwerke
ofthe
distributiongrids
Generation RWE,LEAG Shareby Over1000
EnBW,E.ON, generation: producers
Vattenfall,Others RWE(25.3%),
LEAG(14.9%),
EnBW(9.9%),
E.ON(9.6%),
Vattenfall
(5.6%),Others
(34.7%)
Retail EnBWE.ONRWE Theyhavea Over1000
suppliers Vattenfall 62%sharein retailers
thetotalretail
market
to provide the TSO with a 15-minute resolution forecast
detailing all anticipated activities for the following day [15].
After receiving forecasts from all balancing groups, the TSO
conducts a power-flow study to identify potential grid bot-
tlenecks. A grid bottleneck occurs when certain transmission
lines lack the necessary capacity to accommodate the traded
energy flow. In such cases, adjustments need to be made
to the power allocation through a process known as “re-
dispatch." This involves reducing the generation output of
units located upstream of the bottleneck and activating units
located downstream. The owners of the down-regulated units
are financially compensated for their participation in the re-
dispatchmeasuresandtheunitsupstreamarecompensatedfor
the power curtailment.
III. ANCILLARYSERVICEMARKET:STATUSQUO
A. Frequency support ancillary products
In power systems, uncertainty is always associated with
forecasted generation and demand, as well as technical distur-
bances (e.g., power plant outages, line faults, cyber-attacks).
Therefore, the TSOs are required to maintain active power
reserves available in real-time, and in case of a shortfall,
the TSOs must take remedial actions. In Germany, the TSOs
procure three different types of reserves (FCR, aFRR, and
mFRR),eachreactingatadifferenttimescale,asillustratedin
Fig. 3. Due to the different reaction times, specific generation
units are considered for each of the reserves.
1) Frequency containment reserve (FCR): The FCR is
equivalent to the primary control reserve. It is the initial
control action and is self-activated within a synchronous area.
Following a disturbance, the FCR should be fully available
within 30 seconds and capable of supporting the grid with
100% capacity for at least 15 minutes. The FCR is currently
RFF
+
a�renI
FCR
secivreS
Emerging services
automated
5s 30s 5 min 12.5 min
Fig.3:Balancingreserves(reservesarebidirectional)
being supplied as part of a joint tender, known as FCR co-
operation, by the TSOs of Germany, Austria, the Netherlands,
Belgium, France, and Switzerland and is accessed through a
common market. In 2019, BNetzA introduced amendments
to the FCR market to make it more accessible for small
stakeholders [35]. These changes included the introduction
of the "15-minute criteria," which requires a continuous and
uninterruptedsupplyoftheagreedamountofFCRforatleast
15 minutes during an "alert state". Previously, the minimum
activation time was 30 minutes [36].
Analertstateisdefinedasafrequencydeviationof|∆f|≥
50 mHz for 15 minutes, |∆f| ≥ 100 mHz for 5 minutes,
or instantaneous |∆f| ≥ 200 mHz [37]. If the frequency
deviation is less than 200 mHz, the system should provide a
partialdeliveryforaproportionatelylongerperiod.Itiscrucial
that the supply begins immediately when the deviation occurs
and continues for at least 15 minutes after the alert state is
triggered. The FCR product has the following characteristics:
(i) The FCR demand is determined on a European level,
according to ENTSO-E guidelines.
(ii) FCR is a symmetric product, i.e., upward and downward
reserves are procured together. For example, 1 MW
FCR represents 1 MW of regulation both upward and
downward [37].
(iii) The auction for FCR takes place daily, offering six four-
hour products for the following delivery day. An auction
closes 1 day before the delivery at 8am [38].
(iv) Divisible and indivisible bids are allowed by TSOs.
Indivisible bids have a maximum bid size of 25 MW
in all participating countries. The minimum bid size is 1
MW.Capacitycanbeofferedinstepsof1MWuptothe
prequalified power [39].
(v) The synchronous area Central Europe (CE) consists of
several load-frequency control (LFC) Blocks. Each LFC
Block consists of one or more LFC Areas. The Monitor-
ing Areas, LFC Areas, and LFC Blocks for Germany are
summarized in Table. III.
(vi) The required FCR volume for LFC areas is annually cal-
culatedonthebasisofeachLFCarea’spowergeneration
anddemand.TheFCRvolumescanalsobetradedamong
the LFC areas within an LFC block [38].
(vii) TherampingrateoftheFCRproductisTSOspecificand
is determined at the time of pre-qualification [40].
(viii) As of October 2023, Germany’s estimated demand for
FCR reserves is 573 MW [41], with BESS systems
constituting two-thirds of the German FCR market [42].
2) Frequency Restoration Reserves (FRR): The FRR re-
serve is further classified into two categories: automatic Fre-

TABLEIII:ListofMonitoringAreas,LFCAreas,andLFCBlocksinGermany,adaptedfrom[39]
Country TSO(fullname) TSO(shortname) MonitoringArea LFCAREA LFCBlock
|     | TransnetBWGmbH |     |     |     | TransnetBW |     | TNG |     | TNG |     |     |     |     |     |     |
| --- | -------------- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Germany 50HertzTransmissionGmbH 50Hertz 50HZT 50HZT TNG+TTG+AMP+50HZT+EN+CREOS
|     | TenneTTSOGmbH |     |     |     | TenneTGER |     | TTG |     | TTG+EN    |     |     |     |     |     |     |
| --- | ------------- | --- | --- | --- | --------- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- |
|     | AmprionGmbH   |     |     |     | Amprion   |     | AMP |     | AMP+CREOS |     |     |     |     |     |     |
quency Restoration Reserves (aFRR) and manual Frequency RTE, Terna, and Swissgrid, are connected through the
Restoration Reserves (mFRR), which act as secondary and TERRE platform [49].
| tertiary | frequency | reserves, | respectively. |     | The deployment |     | of  |     |     |     |     |     |     |     |     |
| -------- | --------- | --------- | ------------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
aFRR should reach 100% capacity within 5 minutes. In the B. Non-frequency ancillary products
| case of      | grid imbalances |         | lasting longer | than    | 5 minutes, |          | mFRR |               |      |           |          |      |           |      |           |
| ------------ | --------------- | ------- | -------------- | ------- | ---------- | -------- | ---- | ------------- | ---- | --------- | -------- | ---- | --------- | ---- | --------- |
|              |                 |         |                |         |            |          |      | Non-frequency |      | ancillary | services |      | encompass | a    | range of  |
| is activated | to              | support | aFRR. The      | trading | of aFRR    | reserves |      |               |      |           |          |      |           |      |           |
|              |                 |         |                |         |            |          |      | services      | used | by TSOs   | and      | DSOs | to ensure | grid | stability |
takesplacethroughtheENTSO-EenabledPlatformfortheIn-
|     |     |     |     |     |     |     |     | [32]. While |     | these services | are | primarily | provided |     | by gener- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | -------------- | --- | --------- | -------- | --- | --------- |
ternational Coordination of Automated Frequency Restoration ators, they can also be offered by demand facilities, network
| and Stable  | System     | Operation | (PICASSO)      |            | [43],     | while     | mFRR |            |           |                   |          |             |            |                   |         |
| ----------- | ---------- | --------- | -------------- | ---------- | --------- | --------- | ---- | ---------- | --------- | ----------------- | -------- | ----------- | ---------- | ----------------- | ------- |
|             |            |           |                |            |           |           |      | operations | and       | equipment,        | and      | energy      | storage    | facilities        | [50].   |
| reserves    | are traded | through   | the Manually   |            | Activated | Reserves  |      |            |           |                   |          |             |            |                   |         |
|             |            |           |                |            |           |           |      | Various    | research  | studies           | at the   | European    | level      | [51]–[53],        | are     |
| Initiative  | (MARI)     | [44].     | Due to         | the longer | delivery  | duration, |      |            |           |                   |          |             |            |                   |         |
|             |            |           |                |            |           |           |      | dedicated  | to        | the establishment |          | of a market |            | for non-frequency |         |
| small-scale | BESS       | units     | are unsuitable |            | for FRR   | services. | In-  |            |           |                   |          |             |            |                   |         |
|             |            |           |                |            |           |           |      | ancillary  | products. | These             | projects | aim         | to enhance |                   | TSO-DSO |
stead,synchronousgenerators,flexibleindustrialloads,virtual coordination and establish local non-frequency ancillary ser-
| power plants,   |     | and pump | storage         | systems | are typically | utilized |       |              |       |              |         |        |        |                |     |
| --------------- | --- | -------- | --------------- | ------- | ------------- | -------- | ----- | ------------ | ----- | ------------ | ------- | ------ | ------ | -------------- | --- |
|                 |     |          |                 |         |               |          |       | vice markets |       | based on     | optimal | power  | flow   | [54]. However, | in  |
| [45]. Following |     | are the  | characteristics | of      | FRR products  |          | [41]: |              |       |              |         |        |        |                |     |
|                 |     |          |                 |         |               |          |       | Germany,     | there | is currently | no      | common | market | mechanism      |     |
(i) aFRR and mFRR are procured as separate products in place for procuring non-frequency ancillary services [55].
through PICASSO and MARI market platforms. These services are acquired through bilateral agreements,
(ii) aFRR and mFRR are asymmetrical products, and a sep- with pricing determined through private negotiations and not
| arate | auction | is held | to procure |     | positive | and negative |     |           |     |             |        |                    |     |     |       |
| ----- | ------- | ------- | ---------- | --- | -------- | ------------ | --- | --------- | --- | ----------- | ------ | ------------------ | --- | --- | ----- |
|       |         |         |            |     |          |              |     | disclosed | to  | the public, | making | it non-transparent |     |     | [56]. |
products. 1) Voltage Regulation: Status Quo: According to ENTOS-
| (iii) The | products | shall | have a minimum |     | bid size | of 1 | MW. |     |     |     |     |     |     |     |     |
| --------- | -------- | ----- | -------------- | --- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
E[57]:"VoltageControlreferstothecontrolactionsdesigned
(iv) Daily auctions offer six four-hour aFRR and mFRR to maintain the set voltage level or the set value of Reactive
products for the next delivery day. Power". The following are the key points regarding voltage
| (v) aFRR | auction | closes | at 9 a.m.,  | one | day before | delivery, |     |         |             |          |       |             |     |     |           |
| -------- | ------- | ------ | ----------- | --- | ---------- | --------- | --- | ------- | ----------- | -------- | ----- | ----------- | --- | --- | --------- |
|          |         |        |             |     |            |           |     | control | in Germany. |          |       |             |     |     |           |
| and      | mFRR    | closes | at 10 a.m., | one | day before | delivery. |     |         |             |          |       |             |     |     |           |
|          |         |        |             |     |            |           |     | (i) In  | Germany,    | reactive | power | procurement |     | is  | primarily |
3) Replacement reserves (RR): RR is the last market dedicatedtofulfillingonlysteady-staterequirements[58].
| available | control | reserve      | and is deployed |     | manually.  | Typically |     |                                                            |     |     |     |     |     |     |     |
| --------- | ------- | ------------ | --------------- | --- | ---------- | --------- | --- | ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|           |         |              |                 |     |            |           |     | (ii) Atthetransmissionlevel,voltagestabilityistheresponsi- |     |     |     |     |     |     |     |
| activated | from    | 15 min after | the imbalance   |     | occurrence | and       | can |                                                            |     |     |     |     |     |     |     |
bilityofTSOs,whoarerequiredtomaintaintheirreactive
stayuptohoursuntilthenormaloperationisrestored.TheRR power compensation units [15].
| products | can be | brought | from the | balancing | service | providers |     |                     |     |         |           |     |           |      |           |
| -------- | ------ | ------- | -------- | --------- | ------- | --------- | --- | ------------------- | --- | ------- | --------- | --- | --------- | ---- | --------- |
|          |        |         |          |           |         |           |     | (iii) Additionally, |     | network | operators |     | can enter | into | bilateral |
(suchasgeneratorsordemandfacilities)orexchangedbetween contracts with individual generation plants to procure
the TSOs through the Trans European Replacement Reserve reactive power for system stability [56].
Exchangemarketplatform(TERRE)[46].Currently,Germany
|     |     |     |     |     |     |     |     | (iv) Reactive |     | power can | be sourced |     | through | various | means |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --------- | ---------- | --- | ------- | ------- | ----- |
does not procure any RR product [47]. Following are the [58], including: (i) Capacitor banks (ii) Synchronous
| characteristics |     | of RR product | [47]: |     |     |     |     |              |     |       |              |     |        |         |        |
| --------------- | --- | ------------- | ----- | --- | --- | --- | --- | ------------ | --- | ----- | ------------ | --- | ------ | ------- | ------ |
|                 |     |               |       |     |     |     |     | compensators |     | (iii) | High Voltage |     | Direct | Current | (HVDC) |
(i) The common platform TERRE has 24 trade phases that links (iv) On-load tap changing transformers (OLTC)
are carried out throughout the day. on Extra High Voltage (EHV) or High Voltage (HV)
(ii) The ramping period of a product can range from 0 to transformersides(v)Energystorageunits(vi)Renewable
30 minutes. However, the total activation time for the Energy Sources (vii) Conventional power plants.
product cannot exceed 30 minutes. (v) According to the Demand Connection Code [59], the
(iii) The smallest volume increment in which bids can be relevant TSO has the authority to specify the reactive
submitted is 1 MW. The delivery duration of the product power range for transmission-connected demand facili-
can be 15, 30, or 60 minutes. ties, including large industries and DSOs.
(iv) The minimum size of a bid is 1 MW. In the case of (vi) Currently, there is no established regulatory framework
a divisible bid, there is no maximum limit, but only governing the settlement rules between TSOs and DSOs
technical limits are constraints. In the case of indivisible (alsotransmission-connecteddemandfacilitiesandindus-
bids, specific national rules are implemented [48]. tries) regarding the pricing of reactive power exchange.
CˇEPS,
(v) Only six TSOs at present, namely REE, REN, Ittypicallydependsonindividualbilateralcontracts[58].

(vii) The code also mandates that the power factor of power Mean= € 68.361
| imported | by  | DSOs | (also | transmission-connected |     |     | demand |     |     |     |     |     |     |     |     |
| -------- | --- | ---- | ----- | ---------------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
FCR prices Euro/MW
| facilities | and | industries) |     | must | not fall | below | 0.9 [59]. |     |     |     |     |     |     |     |     |
| ---------- | --- | ----------- | --- | ---- | -------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
(viii) To regulate reactive power, DSOs in Germany have the 4-hourly service period
FCR prices
| flexibility           | to       | choose             | between     | different     |               | fixed      | power       | factor |                     |     |     |     |     |     |     |
| --------------------- | -------- | ------------------ | ----------- | ------------- | ------------- | ---------- | ----------- | ------ | ------------------- | --- | --- | --- | --- | --- | --- |
| control               | methods, | including          |             | cos(ϕ),       | cos(ϕ(P)),    |            | and         | Q(V)   |  WM/oruE secirp RCF |     |     |     |     |     |     |
| droop                 | control, | for                | local       | voltage       | regulation    | [52].      |             |        |                     |     |     |     |     |     |     |
| (ix) In the           | event    | of voltage-related |             |               | issues        | at the     | DSO         | level, |                     |     |     |     |     |     |     |
| corrective            | measures |                    | are         | undertaken    |               | by the     | TSOs        | after  |                     |     |     |     |     |     |     |
| exchanging            |          | necessary          | information |               | [52].         |            |             |        |                     |     |     |     |     |     |     |
| (x) TSO-connected     |          | DSOs               | and         | demand        |               | facilities | are         | penal- |                     |     |     |     |     |     |     |
| ized                  | if they  | fail to            | adhere      | to            | the specified |            | power       | factor |                     |     |     |     |     |     |     |
| regulations           |          | as decided         | in          | the bilateral |               | contracts  | [52].       |        |                     |     |     |     |     |     |     |
| (xi) Power-generating |          |                    | modules     | in            | Germany       | are        | categorized |        |                     |     |     |     |     |     |     |
Year
| based | on voltage | levels | and | installed |     | generation | capacity. |     |     |     |     |     |     |     |     |
| ----- | ---------- | ------ | --- | --------- | --- | ---------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
Fig.4:FCRpricedevelopmentinGermanyfromthebeginningof4-hrservice
Detailed specifications of these four generator types in period(sinceJuly2020),dataretrievedfrom[41]
| Germany        | and | their | respective | reactive |       | power            | capabilities |     |         |        |      |           |         |      |          |
| -------------- | --- | ----- | ---------- | -------- | ----- | ---------------- | ------------ | --- | ------- | ------ | ---- | --------- | ------- | ---- | -------- |
| are summarized |     | in    | Table      | IV and   | Table | V, respectively. |              |     |         |        |      |           |         |      |          |
|                |     |       |            |          |       |                  |              |     | The FCR | prices | have | decreased | by more | than | 50% from |
TABLEIV:Categoriesofpower-generatingmodulesinGermany[60] 2015 to 2021 [13]. Furthermore, the price trends of FCR
|           |                         |     |     |                     |        |     |              |     | reserves, | as shown   | in     | Fig. 4, have | remained     | considerably | low   |
| --------- | ----------------------- | --- | --- | ------------------- | ------ | --- | ------------ | --- | --------- | ---------- | ------ | ------------ | ------------ | ------------ | ----- |
| Generator | Requirements            |     |     | Activepowercapacity |        |     | Voltagelevel |     |           |            |        |              |              |              |       |
|           |                         |     |     |                     |        |     |              |     | for the   | last three | years, | with         | a mean price | of C68.36    | since |
| TypeA     | Capabilitiestowithstand |     |     | 0.8kW≤P             | <135kW |     | <110kV       |     |           |            |        |              |              |              |       |
wide-scaleevents. July 2020 [41]. The low prices can be attributed to falling
Limitedautomated battery prices and market saturation due to the limited FCR
responseandcontrol.
|       |                  |     |     |         |       |     |        |     | volume | of 573 | MW, causing | the | FCR market | to  | become less |
| ----- | ---------------- | --- | --- | ------- | ----- | --- | ------ | --- | ------ | ------ | ----------- | --- | ---------- | --- | ----------- |
| TypeB | Automateddynamic |     |     | 135kW≤P | <36MW |     | <110kV |     |        |        |             |     |            |     |             |
response. profitable over the years [13], [14], [62]. During the third
Systemoperatorcontrol. quarter of 2021, there was a notable surge in FCR prices,
| TypeC | Stableandcontrollable |     |     | 36MW≤P | <45MW |     | <110kV |     |     |     |     |     |     |     |     |
| ----- | --------------------- | --- | --- | ------ | ----- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
dynamicresponse. which can be linked to the Russia-Ukraine conflict and the
Coveringalloperational uncertainty surrounding the import of Russian gas. Fig. 5 and
networkstates. Fig. 6 illustrate the progression of aFRR and mFRR average
| TypeD | Wide-scalenetwork |     |     | P   | ≥45MW |     | ≥110kV |     |     |     |     |     |     |     |     |
| ----- | ----------------- | --- | --- | --- | ----- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
operationandstability. prices in Germany for the year 2022, which is considerably
Balancingservices. low [63]. Research from [64] has indicated that the prevailing
|     |     |     |     |     |     |     |     |     | aFRR prices | in  | Germany | do  | not offer an | economic | case for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------- | --- | ------------ | -------- | -------- |
TABLEV:Reactivepowercapabilityforthegeneratingmodules[61] battery storage participation in the market [41]. Additionally,
theextendeddeliverytimeformFRRexcludestheBESSunits
| ModuleType |     | TypeB |     |     | TypeC,D |     |     |     |     |     |     |     |     |     |     |
| ---------- | --- | ----- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Synchronous Specifiedbythesystem U−Q/Pmaxprofilewith from market participation.
| powermodules |     | operator |     |     | amaximumrangeof0.95 |     |     |     |     |     |     |     |     |     |     |
| ------------ | --- | -------- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
forQ/Pmax
| Power park     | gen- | Specifiedbythesystem |     |     | U−Q/Pmaxprofilewith |     |     |     |     |     |     |     |     |     |     |
| -------------- | ---- | -------------------- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erationmodules |      | operator             |     |     | amaximumrangeof0.75 |     |     |     |     |     |     |     |     |     |     |
forQ/Pmax
|     |     | U−Q/Pmax |     |     | U−Q/Pmaxprofilewith |     |     |     |     |     |     |     |     |     |     |
| --- | --- | -------- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Offshore power profile ]WM[ yticapaC evreseR ]h/)WM /RUE([ ecirP
| parkmodules |     |     |     |     | amaximumrangeof0.75 |     |     |     |     |     |     |     |     |     |     |
| ----------- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
forQ/Pmax
| C. Economics      | of            | Ancillary   | Products          |       | in Germany |                     |     |        |     |     |     |     |     |     |     |
| ----------------- | ------------- | ----------- | ----------------- | ----- | ---------- | ------------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
| According         | to §          | 12h EnWG    |                   | [32]; | BNetzA     | is responsible      |     | for    |     |     |     |     |     |     |     |
| evaluating        | the potential |             | for "transparent, |       |            | non-discriminatory, |     |        |     |     |     |     |     |     |     |
| and market-based" |               | procurement |                   | of    | certain    | services            |     | in the |     |     |     |     |     |     |     |
Month
| energy sector. | If  | market-based |     | procurement |     | is  | found | to be |     |     |     |     |     |     |     |
| -------------- | --- | ------------ | --- | ----------- | --- | --- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
Fig.5:AverageaFRRvolumeandpricesinGermanyfortheyear2022[41]
economicallyinefficientforaparticularservice,BNetzAmust
define an exception and specify an alternative procurement The prices for reactive power are not uniform and can vary
method. Based on these guidelines, the frequency control significantly due to different bilateral contracts between TSOs
reserves are auctioned via a market platform. In the case and generators/consumers. The cost to customers for reactive
of non-frequency ancillary services, only voltage regulation power demand can increase to a maximum of 9.20 C/MVArh
servicesandtheblackstartservicesaredeemedtobeprocured [57].Similarly,powerplantsarecompensatedbytheTSOsfor
through market procedure. However, currently, no market supplying reactive power, but the specific prices are not pub-
mechanism is in place for them [56]. licly disclosed and are determined through individual bilateral

|     |     |     |     |     |     |     | interconnected |             | system   | [69].   | A        | similar  | frequency |           | event on |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- | -------- | ------- | -------- | -------- | --------- | --------- | -------- |
|     |     |     |     |     |     |     | 10 January     | 2019        | revealed |         | that the | grid’s   | capacity  | imbalance |          |
|     |     |     |     |     |     |     | exceeded       | the primary |          | control | reserve  | capacity |           | of 2.6    | GW at    |
]WM[ yticapaC evreseR
|     |     |     |     |     |     |     | the time | [73]. | Another | grid | event | occurred | on  | 14 December |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | ------- | ---- | ----- | -------- | --- | ----------- | --- |
]h/)WM /RUE([ ecirP
|     |     |     |     |     |     |     | 2018, where      | an           | incorrect | forecast |       | for photovoltaic |                | feed-in      | led       |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ------------ | --------- | -------- | ----- | ---------------- | -------------- | ------------ | --------- |
|     |     |     |     |     |     |     | to a grid        | imbalance    | in        | Germany  | [73]. | Germany’s        |                | contribution |           |
|     |     |     |     |     |     |     | to instantaneous |              | (inertia) | reserves |       | is notably       |                | low,         | making it |
|     |     |     |     |     |     |     | vulnerable       | to withstand |           | any      | spilt | event,           | as highlighted |              | by the    |
studiesfromGermanTSOs[74],[75].Therefore,bridgingthe
|     |     |     |     |     |     |     | gap between | fault    | occurrences |         | and       | the   | activation | of     | primary    |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ----------- | ------- | --------- | ----- | ---------- | ------ | ---------- |
|     |     |     |     |     |     |     | frequency   | reserves | is          | crucial | to ensure | power |            | system | stability. |
Month
|     |     |     |     |     |     |     | Voltage | stability | is  | also | facing | significant |     | challenges | as  |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | ---- | ------ | ----------- | --- | ---------- | --- |
Fig.6:AveragemFRRvolumeandpricesinGermanyfortheyear2022[41]
|           |         |             |          |     |       |     | renewable    | energy   | penetration |      | increases. |        | Research    |          | findings |
| --------- | ------- | ----------- | -------- | --- | ----- | --- | ------------ | -------- | ----------- | ---- | ---------- | ------ | ----------- | -------- | -------- |
|           |         |             |          |     |       |     | from project | MIGRATE  |             | [76] | suggest    | that   | the         | existing | voltage  |
| contracts | between | the parties | involved |     | [57]. |     |              |          |             |      |            |        |             |          |          |
|           |         |             |          |     |       |     | control      | measures | alone       | will | no         | longer | be adequate |          | in the   |
IV. INADEQUACYOFPRESENTCONTROLRESERVES future. In Germany, projections suggest a potential reactive
|     |     |     |     |     |     |     | power deficit | of  | 38.1 | GVar | to 74.3 | GVar | by  | 2035 [77]. | This |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---- | ---- | ------- | ---- | --- | ---------- | ---- |
The2023reportfromGermanTSOAmpriononthecurrent
status of frequency reserves [65], along with the analysis deficit arises from the inability of inverter-based generation
|           |        |        |           |           |     |                | to provide | the | same | level | of reactive | power |     | as synchronous |     |
| --------- | ------ | ------ | --------- | --------- | --- | -------------- | ---------- | --- | ---- | ----- | ----------- | ----- | --- | -------------- | --- |
| conducted | by the | Energy | Economics | Institute |     | at the Univer- |            |     |      |       |             |       |     |                |     |
sity of Cologne (EWI) [66], have emphasized the potential generators [78]. Addressing the reactive power requirements
|                        |        |             |               |       |         |            | of wind    | and       | solar | power        | plants | can | be achieved |          | through |
| ---------------------- | ------ | ----------- | ------------- | ----- | ------- | ---------- | ---------- | --------- | ----- | ------------ | ------ | --- | ----------- | -------- | ------- |
| future vulnerabilities |        | to the      | German        | power | system  | associated |            |           |       |              |        |     |             |          |         |
|                        |        |             |               |       |         |            | additional | inverters |       | or a STATCOM |        |     | device.     | However, | this    |
| with low               | levels | of inertia. | Additionally, |       | studies | conducted  |            |           |       |              |        |     |             |          |         |
by European TSOs [67]–[69] have raised concerns about the approach requires oversizing inverters, leading to substantial
|                 |            |            |                 |       |                |               | investment      | costs.  |               |             |           |                |     |                |     |
| --------------- | ---------- | ---------- | --------------- | ----- | -------------- | ------------- | --------------- | ------- | ------------- | ----------- | --------- | -------------- | --- | -------------- | --- |
| dimensioning    | criteria   | of FCR     | reserves        |       | in Continental | Europe,       |                 |         |               |             |           |                |     |                |     |
|                 |            |            |                 |       |                |               | Furthermore,    |         | the           | integration | of        | inverter-based |     | generators     |     |
| which are       | based      | on a 3000  | MW              | power | outage         | assumption    |                 |         |               |             |           |                |     |                |     |
|                 |            |            |                 |       |                |               | results in      | reduced | short-circuit |             | currents  |                | due | to semiconduc- |     |
| [70]. This      | assumption | originates |                 | from  | the 2012       | ad-hoc report |                 |         |               |             |           |                |     |                |     |
|                 |            |            |                 |       |                |               | tor limitations |         | [78].         | During      | transient | events,        |     | the doubly     | fed |
| by the European |            | Network    | of Transmission |       | System         | Operators     |                 |         |               |             |           |                |     |                |     |
forElectricity(ENTSO-E)[70].Giventhesignificantincrease induction generator turbines behave as Squirrel Cage Induc-
|     |     |     |     |     |     |     | tion Generators, |     | consuming |     | reactive | power | and | reducing | the |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --------- | --- | -------- | ----- | --- | -------- | --- |
intheshareofrenewableenergysourcesinEuropesince2012,
|             |     |          |                  |     |          |         | system’s      | voltage | stability | limit   | [78]. | The | 2021 | market | report   |
| ----------- | --- | -------- | ---------------- | --- | -------- | ------- | ------------- | ------- | --------- | ------- | ----- | --- | ---- | ------ | -------- |
| reassessing | and | updating | the dimensioning |     | criteria | for FCR |               |         |           |         |       |     |      |        |          |
|             |     |          |                  |     |          |         | by the German |         | TSO,      | Amprion | [79], | has | also | raised | concerns |
reservesiscrucial.Fig.7illustratesthetrendforpowersystem
|             |                |     |        |         |      |            | about the  | diminishing |      | sources | of          | reactive | power      | support | in       |
| ----------- | -------------- | --- | ------ | ------- | ---- | ---------- | ---------- | ----------- | ---- | ------- | ----------- | -------- | ---------- | ------- | -------- |
| inertia (H) | in Continental |     | Europe | for the | year | 2030 [71]. |            |             |      |         |             |          |            |         |          |
|             |                |     |        |         |      |            | the German | power       | grid | and     | the growing |          | occurrence |         | of rapid |
voltagefluctuations.Toensurevoltagestability,adiverserange
H ≥ 4s
ofvoltageancillaryserviceswillberequiredinthenearfuture,
3s ≤ H ≤ 4s
2s ≤ H ≤ 3s
|     |     |     |     |     |     |     | covering | various | time | scales | from | milliseconds |     | to hours. | The |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ---- | ------ | ---- | ------------ | --- | --------- | --- |
H < 2s
|     |     |     |     |     |     |     | anticipated             | issues | stemming   |           | from        | a substantial |          | increase | in         |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | ------ | ---------- | --------- | ----------- | ------------- | -------- | -------- | ---------- |
|     |     |     |     |     |     |     | power electronics-based |        |            | renewable |             | generation    |          | in the   | power      |
|     |     |     |     |     |     |     | system can              | be     | summarized |           | as follows: |               |          |          |            |
|     |     |     |     |     |     |     | (i) Frequency           |        | stability  |           | concerns:   | The           | decrease |          | in inertia |
resultingfromfewersynchronousgeneratorsinthepower
|     |     |     |     |     |     |     | system       | may        | lead        | to frequency |              | instability, |                | particularly | in        |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | ----------- | ------------ | ------------ | ------------ | -------------- | ------------ | --------- |
|     |     |     |     |     |     |     | the          | event      | of a system |              | split in     | Central      | European       |              | system.   |
|     |     |     |     |     |     |     | (ii) Voltage | stability: |             | Renewable    |              | generators   |                | often        | have lim- |
|     |     |     |     |     |     |     | ited         | capability | for         | voltage      | and          | reactive     | power          | regulation,  |           |
|     |     |     |     |     |     |     | which        | can        | lead        | to local     | voltage      | instability, |                | especially   | at        |
|     |     |     |     |     |     |     | low          | voltage    | levels      | due          | to dispersed |              | PV generation. |              |           |
Fig.7:Expectedinertiacontributionofcountriesfortheyear2030[71]
|     |     |     |     |     |     |     | (iii) Reduced |     | rotor | angle | stability: | The | presence |     | of fewer |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----- | ----- | ---------- | --- | -------- | --- | -------- |
ReducedpowersysteminertialeadstohighRateofChange synchronous generators and faster power system dynam-
of Frequency (RoCoF) values. The current FCR reserves ics can negatively impact rotor angle stability.
cannot address the issue of high RoCoF values due to their (iv) Power oscillations and damping issues: With the grad-
slower response time compared to the inertial response [72]. ualphasingoutofsynchronousgenerators,thepowersys-
The delay in response can lead to unintentional tripping of temstabilizersmayalsobecomelesseffective,potentially
linesandgenerators.Forinstance,asignificantgriddisruption resulting in power oscillations and reduced damping.
occurred in early 2021 in Europe following a sudden 0.25 Hz (v) Harmonicandpowerquality:InverterPWMswitching,
frequency drop, resulting in the separation of the European interactions between power converters and passive com-

Time scale
| Very short |     |     |      Short |     |   Medium  |     |      Long |     |              |     |     |     |
| ---------- | --- | --- | ---------- | --- | --------- | --- | --------- | --- | ------------ | --- | --- | --- |
|            |     |     | ≈ seconds  |     | ≈ minutes |     |   ≈ hours |     |   Stationary |     |     |     |
    ≈ ms
|     | D y n a m | i c   |     |     |     | F r e q u e n c y |     |     |     |     |     |     |
| --- | --------- | ----- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
F a s t  f r e q u en c y F r e q u e n c y G r i d   r e inforcement
r e a c t i v e   r e s t o ra t i o n r e p l a c e m e n t  de f e r r a l
|     | r e sp o n | s e   (DRR) |     | re s e r v e   (F F | R ) | r e s e r v e   ( F | R R) |                    |       |                   |                      |     |
| --- | ---------- | ----------- | --- | ------------------- | --- | ------------------- | ---- | ------------------ | ----- | ----------------- | -------------------- | --- |
|     |            |             |     |                     |     |                     |      | r e s e r v e s  ( | R R ) |                   |                      |     |
|     |            |             |     |                     |     |                     |      |                    |       | L o c             | a l  v o l t a g e   |     |
|     | Harmonic   |             |     | Frequency           |     | Local voltage       |      | Local voltage      |       |                   |                      |     |
|     |            |             |     | containment         |     |                     |      |                    |       | su p              | p o r t  t o   E V   |     |
|     | mitigation |             |     |                     |     | /reactive power     |      | support            |       | charging stations |                      |     |
|     |            |             |     | reserve (FCR)       |     | support             |      |                    |       |                   |                      |     |
Loss Minimization
|     | Fault Current  |     |     | FCR-Normal          |     |                 |     |                 |     |                      |     |     |
| --- | -------------- | --- | --- | ------------------- | --- | --------------- | --- | --------------- | --- | -------------------- | --- | --- |
|     | Injection      |     |     |                     |     | Black start     |     | Black start     |     | with local reactive  |     |     |
|     |                |     |     | FCR-Disturbance     |     |                 |     |                 |     | power support        |     |     |
|     |                |     |     | Low frequency       |     | Inertia through |     |                 |     |                      |     |     |
|     | Virtual        |     |     |                     |     |                 |     | Stable Islanded |     |                      |     |     |
|     | inertia        |     |     | oscillation         |     | thermal load    |     |                 |     | Capacity firming     |     |     |
|     |                |     |     | damping             |     | flexibility     |     | operation       |     |                      |     |     |
|     |                |     |     | Strategic islanding |     | Active power    |     | Congestion      |     |                      |     |     |
|     | Fast frequency |     |     |                     |     |                 |     |                 |     | Co-located           |     |     |
reserve (FFR) /Grid forming ramping management renewables firming
|     | Fault ride  |     |     | Fault ride       |     |     |     | Peak shaving/  |     |     |     |     |
| --- | ----------- | --- | --- | ---------------- | --- | --- | --- | -------------- | --- | --- | --- | --- |
|     | through     |     |     | through (<1.5s)  |     |     |     | load smoothing |     |     |     |     |
Fig.8:OverviewofpotentialgridancillaryservicesinGermanyanticipatedforthenearfuture
ponentsintheACnetwork,leadingtopotentialresonance ancillary products operating at different time resolutions: (i)
problems and harmonic magnification. Dynamic Containment, (ii) Dynamic Moderation, and (iii)
|     |     |     |     |     |     | Dynamic | Regulation. | The | idea behind | dynamic | products | is to |
| --- | --- | --- | --- | --- | --- | ------- | ----------- | --- | ----------- | ------- | -------- | ----- |
V. ANCILLARYPRODUCTS:FUTUREREQUIREMENTS
|     |     |     |     |     |     | counter | reduced | system inertia | and | help restore | the frequency |     |
| --- | --- | --- | --- | --- | --- | ------- | ------- | -------------- | --- | ------------ | ------------- | --- |
The preceding sections have presented the current status faster.Table.VIIsummarizestheFFRproductsfromdifferent
of grid ancillary services in Germany and have highlighted TSOs around the world, along with the product specifica-
the inadequacies of these services in supporting the existing tions.The IEEE standard P2800-2022 [87] has provided the
German power system infrastructure as it prepares for an classification and technical specifications for FFR services.
| increased | share of | renewables | in the | near future. | This section |     |     |     |     |     |     |     |
| --------- | -------- | ---------- | ------ | ------------ | ------------ | --- | --- | --- | --- | --- | --- | --- |
However,theimplementationofFFRservicescanbechalleng-
shiftsthefocustotheanticipatedneedsandemergingsolutions ing because during an event of fault, there can exist different
for enhancing the resilience of German power systems. Fig. 8 local frequencies and different Rates of Change of Frequency
shows the complete list of potential ancillary services. The (RoCoF) at different buses [72]. Therefore, this may lead to
services are classified based on the required delivery time false triggering of services.
scales. The section begins with examining future frequency 2) Virtual inertia services: In the future, the responsibility
support ancillary products, offering a rationale for why these of providing virtual inertia will be assumed by inverters [88].
services may be needed in Germany, and showcasing imple- However, the market and incentives for virtual inertia in the
mentationexamplesofsimilarservicesfromvariouscountries. ancillary services market are currently lacking [89]. The Ger-
Subsequently, the future non-frequency ancillary products are man network development plan for 2021-2035 has introduced
explained in detail. inertia requirement guidelines for the TSOs [29]. In response
|              |         |           |          |     |     | to these | guidelines,  | the German |     | TSOs are   | already    | working |
| ------------ | ------- | --------- | -------- | --- | --- | -------- | ------------ | ---------- | --- | ---------- | ---------- | ------- |
| A. Frequency | support | ancillary | products |     |     |          |              |            |     |            |            |         |
|              |         |           |          |     |     | towards  | establishing | a separate |     | market and | incentives | for     |
1) FastFrequencyReserves(FFR): FFRreferstotherapid virtual inertia to ensure its effective integration and utilization
adjustment of active power, involving both generation and within power systems [90]. Virtual inertia and FFR are two
load, within a short time frame (typically 2 seconds or less) differentancillaryproducts,accordingtotheAustralianEnergy
| [80]. The | significance | of  | FFR services | in ensuring | stability |        |          |        |       |     |     |     |
| --------- | ------------ | --- | ------------ | ----------- | --------- | ------ | -------- | ------ | ----- | --- | --- | --- |
|           |              |     |              |             |           | Market | Operator | (AEMO) | [85]: |     |     |     |
in low inertia power systems has been emphasized in the "Inertia provides an inherent response to slow the RoCoF,
report by the North American Electric Reliability Corporation but cannot act to restore power system frequency. ̸= FFR
(NERC) [81]. ENTSO-E report on future inertia [82], have can inject active power to correct the imbalance and restore
estimated the average power system inertia for continental system frequency, but does not inherently slow RoCoF in the
| Europe | as illustrated | in  | Fig. 7. | To address | the challenges | same | manner." |     |     |     |     |     |
| ------ | -------------- | --- | ------- | ---------- | -------------- | ---- | -------- | --- | --- | --- | --- | --- |
posedbylowinertiaphases,theNordiccountrieshavealready The Australian Energy Market Commission (AEMC) has
implementedFFRasanancillaryproduct[83].Batterystorage requestedtheestablishmentofaninertiaspotmarkettoensure
systems and demand side response are usually utilized for the grid stability [91]. The release of the procurement mecha-
FFR services [84], [85]. Similar to the FFR product of the nism draft is expected by February 2024. Similarly, Great
Nordic grid, Great Britain has also introduced a new suite of Britainisalsomovingtowardsintroducinginertiaprocurement
dynamic response services [86], that includes three dynamic through bilateral contract mechanisms. During the COVID-19

OscillationSource Location Frequency(Hz)
pandemic, the electricity system in Great Britain encountered
LCC-HVDC Yongfu,Guangxi,China 3
difficultiesduetothesignificantpresenceofrenewableenergy WindFarm Texas,USA(ERCOT) 4
sourcescoupledwithreducedelectricaldemand.Thissituation WindFarm Jilin,Tongyu,China 3-6
WindFarm Hornsea,U.K 8.5
compromised the stability of the grid due to a lack of inertia.
MMC-HVDC Fujian,Xiamen,China 23.6-25.2
To ensure stability, the British TSO had to secure substantial MMC-HVDC Nanhui,Shanghai 20-30
capacity reserves. However, this excessive compensation re- WindFarm Xinjiang,China 30
BESS Moloka,Hawai 18
sultedinathree-foldincreaseinthecostofprocuringancillary
services during the months of May to July 2020 compared to TABLEVI:Poweroscillationeventsobservedaroundtheworld[101]–[103].
the same period in 2019 [92]. To address the high costs of
Modeling Task Force [105] has identified the potential of
ancillary services, a real-time inertia monitoring framework
Battery Energy Storage Systems (BESS) in providing damp-
has now been established, along with the introduction of
ing ancillary services. Additionally, research [106]–[108] has
bilateral virtual inertia contracts [93]. These contracts, valued
demonstrated the effectiveness of inverters in damping LFOs
at £323 million, are scheduled to begin in April 2024. The
and ensuring the stability of microgrids. Currently, there is
virtual inertia is allocated between synchronous condensers
noestablishedmarketforelectromechanicaldampingservices.
and grid-forming converters, with the latter representing a
However, a report by NREL [109] advocates for introducing
pioneering large-scale implementation [93], [94].
battery storage systems to provide damping services in the
3) Frequency containment reserve enhancement products:
United States. Similarly, research by [110] highlights the
AccordingtoajointpaperbythefourEuropeanTSOs:Swiss-
impactofincreasingwindpenetrationonsustainedoscillations
grid,Amprion,Stattnet,andRedEléctricadeEspaña[68],the
in the Irish grid and recommends incentivizing generators
establishment of energy markets and extensive renewable en-
to offer damping services. In Germany, with plans to nearly
ergygenerationcouldpotentiallyresultininsufficientvolumes
double its onshore wind capacity to 115 GWh by 2032 and
of FCR. Therefore, it may be advantageous to consider the
expandtheHVDCcorridornetwork[111],theroleofdamping
re-dimensioning of current FCR reserves and the introduction
services will become even more crucial.
of new FCR products on top of it. For example, the Nordic
5) Ramping margin services: The ramping margin refers
TSOs realized the insufficiency of the FCR volumes in the
to a power unit’s predetermined additional capacity to of-
Nordic synchronous area [95]. Driven by the need to handle
fer the system operator within a particular time frame and
stochastic imbalances and sudden disturbances arising from
duration [5]. With the growing share of variable renewable
the decreasing inertia conditions [95], the Nordic TSOs have
energy, the net load curve, commonly called the duck curve,
introduced two separate FCR products: FCR-N, designed for
becomes more volatile. This necessitates the availability of
normal operating conditions with a volume of 600 MW, and
reserves that can rapidly respond to the fluctuations in net
FCR-D,intendedforlargedisturbanceswithavolumeof1450
load. Traditionally, the responsibility of meeting net load
MW. The overall FCR volume was increased from 1450 MW
ramping requirements has fallen on conventional generators
to approximately 2050 MW (FCR-N + FCR-D). Adopting
[116]. However, ramping services provided by conventional
a similar approach in Continental Europe can lead to an
generators are not recognized as separate ancillary services
increase in the FCR dimension for stability enhancement and
and are only compensated based on the marginal cost of
the expansion of the market volume. Moreover, this approach
electricity production. When the ramping procured through
createsopportunitiesforsmall-scalebatterystoragesystemsto
energy markets is insufficient to handle the increasingly steep
participateinthemarket.Byleveragingthedynamicparameter
and uncertain ramps caused by VRE, the introduction of
tuning, these BESS units can efficiently provide both FCR-N
additional ramping products becomes necessary. Fig. 9 shows
and FCR-D products [96], opening more revenue streams.
thecomparisonintheshapeofthenetloadcurvefrom2016to
4) Damping power system oscillations: The introduction
2023 [117], it can be realized that: (i) there are steep changes
of HVDC interconnections, offshore wind farms, and power
in net load profile, (ii) there is a sharp bidirectional change in
electronic devices, combined with weak grid conditions, is
the net load (iii) negative net load.
leading to an increased occurrence of power oscillations
globally. Some of these incidents are summarized in Table
40
VI. For example, integrating offshore wind power plants has 2016 2023
resulted in subsynchronous oscillations at 4 Hz in the Texas
30
grid and 30 Hz in China [97]. Traditionally, Power System
Sharp ramping
Stabilizers (PSSs) have been used in synchronous generators bi-direc�onal
tomitigatepoweroscillations[98].However,withthegradual 20
phase-out of synchronous generators, the responsibility for
damping these oscillations is shifting to Inverter-Based Re- 10
sources (IBRs) [99], [100]. Nega�ve
net load Sharp ramp
Reports from the MIGRATE project [104] have also high- 0
12 am 12 am
lighted the urgent need to address Low-Frequency Oscilla-
tions (LFOs) in the network. The WECC Renewable Energy
)WG(
daol
teN
Hour of day
Fig.9:Germany’snetloadcurvecomparisonforaspecificday(April10)for
theyear2016and2023,adaptedfrom[117]

TABLEVII:FFRproductsandspecificationsindifferentcountries
ONS CEN EirGrid ERCOT NationalGrid AEMO
Parameters NordicTSOs[115]
(Brazil)[112] (Chile)[112] (Ireland)[52] (Texas)[113] (UK)[86] (Australia)[114]
<0.5second 1.3s(f<49.7Hz)
0.5s(f<59.7Hz)
Responsetime <1second <1second <2second <0.5second <1second 1.0s(f<49.6Hz)
0.25s(f<59.85Hz)
<2second 0.7s(f<49.5Hz)
>15minutes
Product >1hour
>5second >5minutes >8second >30minutes >6second >5second
duration >15minutes
>60minutes
Based on the analysis of Germany’s net load curve, it dips in excess of 30% that would achieve at least a reactive
is imperative to introduce an independent ramping margin power in MVar of 31% of the registered capacity at nomi-
product within the German electricity market. Currently, in nal voltage. The reactive current response shall be supplied
Germany, the provision of ramping services is handled by with a rise time no greater than 40 ms and a settling time
the aFRR and mFRR reserves, with the TSO defining the less than 300 ms. BESS presents an excellent potential for
ramping rate during the pre-qualification process [53], [40]. providing dynamic reactive power response [5], [122]. The
Recognizing this necessity, several system operators in the 2021 market report from the German TSO, Amprion, has
USA, including CAISO (2015), MISO (2011), and EirGrid, highlighted the immediate need for dynamic reactive power
have taken steps to enhance their energy and reserves markets services in German power systems to encounter the frequent
by incorporating three different ramping products [53], [118]. incidents of rapid voltage fluctuations [79]. Fig. 10 shows the
6) Grid forming capability: Findings from project MI- fluctuations in transmission voltages due to high renewable
GRATE [76] and study from German TSO Amprion [75] intermittency and the inability of existing voltage support
provide evidence that future system splits, with a renewable measures to ensure voltage stability. Great Britain has been
energy share of 60 to 70%, cannot be effectively managed a trailblazer in introducing DRR ancillary services through
using the currently implemented grid-following inverters in marketmechanisms.IndividualDERsandvirtualpowerplants
the German grid. In response to this challenge, the German can participate in the market to provide DRR services [123].
TSOAmprionemphasizestheurgentneedforintegratinggrid- Similarly, EirGrid, the Irish TSO, has also included DRR as
forminginvertersintotheGermanpowersystems[75].Ajoint an ancillary service [118], [124].
paper released by the four German TSOs further emphasizes 2) Voltage harmonics mitigation: Power electronics-based
the necessity of deploying grid-forming inverters to ensure nonlinear loads within the distribution system have become
stablegridoperation,especiallywithahighshareofconverter- a significant issue for power systems operators, leading to
based generation ( > 60%) [74]. In addition, the European concerns regarding power quality. These loads can introduce
Union–funded project OSMOSE [119] recommends including voltage distortion, adversely impacting the sensitive loads and
grid-formingcapabilityrequirementsforallnewtransmission- interfering with communication systems. Traditional methods
connected batteries in the European grid. for mitigating harmonics involve passive shunt filters that
The Energy Systems Integration Group (ESIG) identifies absorb specific harmonics within a defined frequency range.
BESS as "low hanging fruits" due to the significant capacity However, these filters are not effective under dynamic operat-
already installed in existing power grids and easy retrofitting ing conditions of a microgrid. An alternative solution is using
to grid forming control [120]. In a similar line, NREL’s active filters, which leverage power electronics technologies
recent report [4] recommends the installation of grid-forming to produce specific current components that cancel out the
technology in new projects and retrofitting old BESS units harmonicscausedbynonlineardevices.Itisfeasibletoprovide
with grid-forming controls. In Great Britain, the National active filter operation as an ancillary service [125]. For the
Grid ESO [121] suggests offering grid-forming capabilities harmonic mitigation market to work, DSOs and TSOs must
as a paid service (and not mandatory) to encourage greater adopt a more flexible approach, and system operators should
adoption of grid-forming inverters in future power systems enabletheimplementationofadaptablegridcodesspecifically
[94]. Considering the concerns raised by German TSOs and at the grid locations where these services are required.
theexamplesofgrid-forminginvertertechnologyandmatured 3) Fault ride through (FRT): Traditionally, power
grid connection specifications from other countries, Germany electronics-based generation units were not designed to
shall consider deploying new renewable projects and large- provide fault ride-through support and would disconnect
scale BESS units with grid-forming control as a proactive from the grid during voltage sags or faults. TSOs anticipate
measure to ensure stable grid operations in the near future. a decrease in short-circuit power as renewable energy
penetration grows. Consequently, there will be a greater
B. Non-frequency ancillary products
occurrence of voltage dips resulting from short-circuit events,
1) Dynamic Reactive Response (DRR): In the initial phase which will affect a larger portion of power generation
of fault or sudden change in renewable generation, it is connected through power electronics. This situation increases
required to deliver a reactive current response for voltage the risk of under-voltage protection tripping for such

|     |     |                   | Hanekenfähr |     |     |     |                    | Bürstadt    |     |     |     | Rommerskirchen    |     |     |     |
| --- | --- | ----------------- | ----------- | --- | --- | --- | ------------------ | ----------- | --- | --- | --- | ----------------- | --- | --- | --- |
|     |     |                   | Hoheneck    |     |     |     |                    | Weißenthurm |     |     |     |                   |     |     |     |
|     |     | Mon., 16 November |             |     |     |     | Thur., 19 November |             |     |     |     | Mon., 23 November |     |     |     |
415
Vk ni egatloV
410
405
400
5:45              6:00               6:15 5:45              6:00               6:15 5:45              6:00               6:15
Fig. 10: Voltage fluctuations at different places in German transmission network (control area Amprion) due to high renewable intermittency and limited
reactivepowersupport(year2021)[79]
generation sources [104]. To avoid those problems, these been procured through bilateral negotiations and contracts
inverter-based units are expected to continue delivering withconventionalplants,suchaspumped-storage,naturalgas-
power to the host grid during faults or voltage sags, ensuring fired, and occasionally coal-fired power plants. These black
uninterrupted power supply to local microgrid loads and start services are primarily compensated for their availability,
offering reactive power support to the grid. This change in enabling them to recover specific investments, operational
perspective highlights how FRT can serve as an additional costs, and maintenance expenses. In some cases, TSOs may
ancillary product at both the TSO and DSO level [126]. provide warming costs to ensure the units remain available
ratherthanbeingshutdown.Accordingto§12h(9)ofEnWG
| 4) Fast   | Post Fault     | Active | Power  | Recovery         |     | (FPFAPR): |     |             |         |                |     |      |     |         |         |
| --------- | -------------- | ------ | ------ | ---------------- | --- | --------- | --- | ----------- | ------- | -------------- | --- | ---- | --- | ------- | ------- |
|           |                |        |        |                  |     |           |     | [32], under | certain | circumstances, |     | TSOs | can | require | storage |
| Following | a transmission |        | fault, | if a significant |     | number    | of  |             |         |                |     |      |     |         |         |
generators fail to recover their rated active power output, it operators to maintain the black start capability of their plants
andbecompensatedaccordingly.ThereportfromtheGerman
| can lead | to a substantial |     | power imbalance, |     | resulting | in  | severe |     |     |     |     |     |     |     |     |
| -------- | ---------------- | --- | ---------------- | --- | --------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
frequency instability. To ensure system stability, power plants EnergyAgency(DENA)[127]recommendsreinforcingplants
|               |     |           |          |           |       |          |       | with black  | start | capabilities | and | introducing |         | an ancillary | ser-      |
| ------------- | --- | --------- | -------- | --------- | ----- | -------- | ----- | ----------- | ----- | ------------ | --- | ----------- | ------- | ------------ | --------- |
| must restore  | at  | least 90% | of their | pre-fault |       | active   | power |             |       |              |     |             |         |              |           |
|               |     |           |          |           |       |          |       | vice market | for   | procurement. |     | Procuring   | through | the          | ancillary |
| output within | 250 | ms after  | the      | voltage   | level | recovers | to at |             |       |              |     |             |         |              |           |
least 90% of its pre-fault value, following the clearance of service market can offer large-scale battery storage facilities
theopportunitytoparticipate[128].ERCOT,theTexassystem
| any fault   | disturbance | within | 900        | ms. Additionally, |            | generators |     |           |             |     |        |             |     |       |              |
| ----------- | ----------- | ------ | ---------- | ----------------- | ---------- | ---------- | --- | --------- | ----------- | --- | ------ | ----------- | --- | ----- | ------------ |
|             |             |        |            |                   |            |            |     | operator, | established | an  | annual | competitive |     | black | start market |
| must remain | connected   | to     | the system | for               | 15 minutes | after      | the |           |             |     |        |             |     |       |              |
|             |             |        |            |                   |            |            |     | in 2009   | [129].      |     |        |             |     |       |              |
faultoccurrence.Centrallydispatchedsynchronousgenerators,
| including | coal, CHP, | biomass, | hydro, |     | pumped | hydro | units, |         |          |      |         |     |            |            |     |
| --------- | ---------- | -------- | ------ | --- | ------ | ----- | ------ | ------- | -------- | ---- | ------- | --- | ---------- | ---------- | --- |
|           |            |          |        |     |        |       |        | 6) Peak | shaving: | Peak | shaving | is  | a strategy | industrial | and |
energystorage,synchronouscompensators,large-scalebattery
commercialpowerconsumersemploytosmoothoutthespikes
units, and wind turbines can all provide this service. EirGrid, intheirelectricityusage.Byimplementingpeakshavingtech-
| the Irish | TSO, has | included | FPFAPR | as  | an ancillary |     | service |         |                 |     |        |         |        |            |     |
| --------- | -------- | -------- | ------ | --- | ------------ | --- | ------- | ------- | --------------- | --- | ------ | ------- | ------ | ---------- | --- |
|           |          |          |        |     |              |     |         | niques, | these consumers |     | aim to | achieve | a more | consistent | and |
since 2014 [124]. Similarly, in Great Britain, the dynamic manageable energy consumption pattern [130]. The primary
| response      | suite has | incorporated | an     | identical | service | for | post- |             |         |          |     |            |           |         |          |
| ------------- | --------- | ------------ | ------ | --------- | ------- | --- | ----- | ----------- | ------- | -------- | --- | ---------- | --------- | ------- | -------- |
|               |           |              |        |           |         |     |       | objective   | is to   | minimize | the | occurrence |           | of high | peaks in |
| fault current | injection | service      | scheme | [86].     |         |     |       |             |         |          |     |            |           |         |          |
|               |           |              |        |           |         |     |       | electricity | demand, | which    | can | strain     | the power | grid    | and lead |
5) Black Start Capability: The black start capability refers to inefficiencies or even blackouts. The EU Electricity Market
to initiating electricity generation without relying on an ex- Revision Survey [131] has suggested introducing a peak
ternal power supply from the electricity grid. In order to shavingancillaryservicethatsystemoperatorscanpurchaseto
provide black start services in Germany, the power plant encourage demand reduction and energy shifting during peak
shall be connected to the transmission grid [58]. Currently, times. Likewise, the European TSOs have also requested the
conventionalplantsprimarilyprovidetheblackstartcapability introduction of peak shaving as a flexibility product [132],
[127]. There are 174 black-start capable plants in Germany [133]. This product can be auctioned with a capacity payment
with a minimum rated output of 10 MW [128]. Accord- andactivatedduringpeakloadwithanenergypayment[131].
ing to Kyon Energy GmbH, the estimated market size for China is leading the way forward in establishing the market
black start capacities is C7.4 million. Additionally, in 2019, for non-frequency ancillary services. Various provinces in
German TSOs contracted a capacity of 4.6 GW for black Chinahavealreadyintroducedpeakshavingancillaryproducts
start reserves [128]. Traditionally, black start capability has throughaspotmarketmechanism[134],[135].Theprovinces

of Fujian, Gansu, Shandong, Jiangsu, Xinjiang, and Guangxi involve both TSOs and DSOs in mitigating congestion and
have carried out peak shaving ancillary products through the maintaining the grid’s stability.
spot market scheme [136]. AccordingtoareportfromDeutscheBank[144],thecostof
7) Arbitrage services: Although arbitrage services are not managinggridcongestionthroughvariousmeasuresamounted
includedinancillaryservices,stackingotherancillaryservices to C771 million in the second quarter of 2023, significantly
with wholesale arbitrage market participation can improve higher than the C389 million recorded in the same period
the profitability of BESS investments [137], [138] [139]. The of 2022. One notable limitation of the current re-dispatch
fundamental principle of power price arbitrage is to exploit framework is the exclusion of storage systems, electric cars,
the variability of electricity prices in the trading market for and demand-side potential as viable options for congestion
financial gain. This strategy involves purchasing electricity management [145]. These resources possess flexibility that
when prices are low and storing it for later sale when prices could aid in alleviating congestion and stabilizing the grid.
increase, resulting in a profit. The profitability of power price However, due to the cost-based approach employed in re-
arbitrage depends on the difference between the lowest and dispatching, which primarily focuses on evaluating fuel and
highest points during a given period. The frequent occurrence labor costs, grid operators face challenges in accurately as-
ofnegativeelectricitypricesinGermanycansupporttheprof- sessing the dispatch costs associated with storage resources.
itability of the arbitrage business model. Negative electricity Consequently, there is a hesitancy to utilize storage systems
pricesofferanopportunitytopurchaseelectricityatextremely due to uncertainties in cost forecasts [145]. Furthermore,
low or even negative costs (being paid for storing the power) according to Amprion, huge power imbalances are expected
[140]. In the German market, the introduction of 15-minute between the north and south of Germany by the year 2030
trade blocks in the continuous intraday market has enhanced [146], as illustrated in Fig. 11.
theflexibilityforbalancingagentstoadjusttheirpositionsand
| take advantage |                                 | of arbitrage | opportunities |     | [15]. |     |     |     |     |     |     |     |
| -------------- | ------------------------------- | ------------ | ------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
| VI.            | INVESTMENTDEFERRALWITHBATTERIES |              |               |     |       |     |     |     |     |     |     |     |
Theincreasingshareofrenewablepowersources(RES)and
| the additional |      | delays in   | Germany’s |      | planned  | grid expansion  |      |     |     |     |     |     |
| -------------- | ---- | ----------- | --------- | ---- | -------- | --------------- | ---- | --- | --- | --- | --- | --- |
| projects       | pose | challenges  | for       | TSOs | in their | day-to-day      | grid |     |     |     |     |     |
| operation      | and  | operational | planning  |      | [141].   | TSOs frequently |      |     |     |     |     |     |
needimmediateremedialactions,suchasre-dispatching,toal-
leviategridcongestion.Re-dispatchmeasuresareindependent
| of the balancing |         | services. | All         | generation | or           | storage facilities |          |     |     |     |     |     |
| ---------------- | ------- | --------- | ----------- | ---------- | ------------ | ------------------ | -------- | --- | --- | --- | --- | --- |
| that a TSO       | obliges | to adjust | their       | active     | power        | for re-dispatch    |          |     |     |     |     |     |
| are remunerated  |         | based     | on reported | costs      | [142].       | However,           | in       |     |     |     |     |     |
| situations       | with    | high RES  | feed-ins,   | the        | availability | of                 | suitable |     |     |     |     |     |
Fig.11:ExpectedpowerimbalanceinGermanyintheyear2030[146]
| power plants | for | re-dispatching |     | is often | insufficient | [52]. |     |     |     |     |     |     |
| ------------ | --- | -------------- | --- | -------- | ------------ | ----- | --- | --- | --- | --- | --- | --- |
Consequently, curtailments of renewable energy feed-ins The launch of the "re-dispatch 3.0" framework [147] in the
are implemented as emergency measures in accordance with nearfuturemayalleviatethecongestionproblemsbyallowing
|      |        |                 |     |         |     |             | producers | and consumers |     | with nominal | ratings | below 100 |
| ---- | ------ | --------------- | --- | ------- | --- | ----------- | --------- | ------------- | --- | ------------ | ------- | --------- |
| EnWG | [143]. | The curtailment |     | process | is  | implemented | se-       |               |     |              |         |           |
quentially, starting with the reduction in RES connected to kW to participate in a re-dispatch market. This inclusion will
|                  |     |       |          |     |               |     | enable the | involvement | of  | electric cars, | home | battery storage, |
| ---------------- | --- | ----- | -------- | --- | ------------- | --- | ---------- | ----------- | --- | -------------- | ---- | ---------------- |
| the transmission |     | grid, | followed | by  | RES connected |     | to the     |             |     |                |      |                  |
distribution grid. This process involves the TSO requesting and heat pumps in re-dispatch measures. Although the market
the DSO to curtail the electricity feed-in. Unlike re-dispatch design for re-dispatch 3.0 is not yet established, TSOs are
|           |               |     |       |            |          |     | working  | on potential | concepts | through | research | projects and |
| --------- | ------------- | --- | ----- | ---------- | -------- | --- | -------- | ------------ | -------- | ------- | -------- | ------------ |
| measures, | a curtailment |     | is an | unbalanced | approach |     | that ne- |              |          |         |          |              |
cessitates additional balancing measures, including balancing field trials [148]. Nevertheless, introducing the "re-dispatch
power. As a result, compensational payments are made for 3.0" framework can potentially create significant economic
|     |     |     |     |     |     |     | opportunities | for battery | asset | owners | [55]. |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ----------- | ----- | ------ | ----- | --- |
curtailingRES,alongwithpaymentsfortherequiredbalancing
| energy. This  | situation | results    |             | in double | payments | [52], | [27].                         |      |         |           |                        |          |
| ------------- | --------- | ---------- | ----------- | --------- | -------- | ----- | ----------------------------- | ---- | ------- | --------- | ---------------------- | -------- |
|               |           |            |             |           |          |       | B. Transmission               | grid | upgrade | deferral: | Grid                   | boosters |
| A. Congestion |           | relief and | re-dispatch |           |          |       |                               |      |         |           |                        |          |
|               |           |            |             |           |          |       | Atotalgridreinforcementcostof |      |         |           | C198billionisestimated |          |
In Germany, the issue of curtailments in the electricity grid for the whole transmission network in Germany by 2037
has resulted in significant compensation claims, amounting to [144]. Large-scale battery storage systems have also gained
approximately C761.2millionin2020.Anewframework,"re- recognition as an effective solution for managing congestion
dispatch2.0,"wasimplementedinresponsetotheselosseson in the power grid and avoiding the re-dispatch costs at the
1 October 2021. This updated regulatory framework allows transmission level. The success of this approach is evident
smaller generating plants with a capacity of 100 kW or more from the installation of three large batteries by the French
to participate in the re-dispatch model. This change aimed to Transmission System Operator (RTE) at the sub-transmission

grid level [149]. In Germany, the unbundling regulations for distribution grids, highlighting that the strategic placement
electricity storage facilities have been in practice since 31 of battery storage systems provides a promising solution to
December 2020, prohibiting system operators from owning defer the expensive upgrades of distribution grids below 50%
the storage facilities. However, in some cases, the network penetration scenario. In summary, by strategically locating
operators may own energy storage facilities if these plants storage units, the grid operators can effectively manage peak
qualify as fully integrated network components [150]. The demand, optimize grid utilization, and minimize the need for
German TSOs have collaboratively proposed the implementa- costly infrastructure investments.
| tion of network |     | boosters | as pilot | projects |     | in their | draft for the |     |     |     |     |     |     |     |
| --------------- | --- | -------- | -------- | -------- | --- | -------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
2030networkdevelopmentplan[7].AmprionTSOiscurrently
|              |                 |               |        |           |           |               |               | VII.             | CONCLUSIONANDRECOMMENDATIONS |          |           |                |             |                |
| ------------ | --------------- | ------------- | ------ | --------- | --------- | ------------- | ------------- | ---------------- | ---------------------------- | -------- | --------- | -------------- | ----------- | -------------- |
| constructing | a               | decentralized |        | network   | booster   | in            | the Bayrisch- |                  |                              |          |           |                |             |                |
|              |                 |               |        |           |           |               |               | The              | paper                        | analyzed | Germany’s | current        | electricity | market         |
| Schwaben     | area,           | which         | will   | utilize   | batteries | with          | a capacity    |                  |                              |          |           |                |             |                |
|              |                 |               |        |           |           |               |               | structure,       | focusing                     | on       | current   | grid ancillary | services,   | procure-       |
| of 250 MW.   | This            | project       | is     | scheduled | to        | become        | operational   |                  |                              |          |           |                |             |                |
|              |                 |               |        |           |           |               |               | ment mechanisms, |                              | and      | pricing   | of services.   | It          | also commented |
| in 2025      | [7]. Similarly, |               | Tennet | TSO       | GmbH      | is developing | two           |                  |                              |          |           |                |             |                |
pilot projects in Audorf South and Ottenhofen, each with a on the suitability of BESS units for providing those services.
|                 |        |     |                 |              |     |          |             | The study | also     | sheds | light       | on the limitations |     | of the existing |
| --------------- | ------ | --- | --------------- | ------------ | --- | -------- | ----------- | --------- | -------- | ----- | ----------- | ------------------ | --- | --------------- |
| capacity        | of 250 | MW, | to be connected |              | to  | the grid | before 2034 |           |          |       |             |                    |     |                 |
|                 |        |     |                 |              |     |          |             | ancillary | products | in    | effectively | addressing         |     | the challenges  |
| [7]. TransnetBW |        | TSO | has             | also planned |     | a 250    | MW battery  |           |          |       |             |                    |     |                 |
project near Kupferzell and Höpfingen and is scheduled to posed by integrating renewable energy sources into the grid
whilesuggestingthenewsetofancillaryservicesthatmaybe
| commence | operations |     | by 2025 | [23]. |     |     |     |     |     |     |     |     |     |     |
| -------- | ---------- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
requiredinthenearfuture.Italsocommentedonthesuitability
C. Distribution grid upgrade deferral of BESS units for providing those services across various
|        |          |        |     |             |         |     |             | time scales, | highlighting |     | their | strong potential |     | as contributors. |
| ------ | -------- | ------ | --- | ----------- | ------- | --- | ----------- | ------------ | ------------ | --- | ----- | ---------------- | --- | ---------------- |
| In the | European | Union, | a   | significant | portion | of  | low-voltage |              |              |     |       |                  |     |                  |
power lines, approximately 35-40%, were between 20 to 40 LiteraturefindingsindicatethatGermansystemoperatorshave
|           |          |       |        |     |          |     |               | acknowledged |     | these | challenges | and are | actively | advocating |
| --------- | -------- | ----- | ------ | --- | -------- | --- | ------------- | ------------ | --- | ----- | ---------- | ------- | -------- | ---------- |
| years old | in 2020, | while | 25-35% |     | exceeded | 40  | years in age, |              |     |       |            |         |          |            |
according to a report from the International Energy Agency for the inclusion of new ancillary services. This necessitates
|              |           |              |                |        |     |          |             | decisive | action     | from     | regulatory | agencies         | to        | collaborate with |
| ------------ | --------- | ------------ | -------------- | ------ | --- | -------- | ----------- | -------- | ---------- | -------- | ---------- | ---------------- | --------- | ---------------- |
| (IEA) [151]. |           | This aging   | infrastructure |        |     | presents | a pressing  |          |            |          |            |                  |           |                  |
|              |           |              |                |        |     |          |             | system   | operators, | identify |            | future ancillary | services, | and es-          |
| issue for    | Germany’s | distribution |                | grids. | A   | report   | by Deutsche |          |            |          |            |                  |           |                  |
Bank [144] underscores the necessity of comprehensive grid tablish profitable procurement mechanisms. The conducted
|     |     |     |     |     |     |     |     | literature | review | has | led to | the following | recommendations: |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | --- | ------ | ------------- | ---------------- | --- |
reinforcementtoensurereliableoperations.Thisreinforcement
is estimated to cost C47 billion by 2037. Similarly, a study 1) Re-dimensioning of control reserves: The size of fre-
from the German Ministry of Economics and Technology quency control reserves in the Continental European grid
[152] suggests that an annual investment ranging from C4.0 shall be re-evaluated and increased to accommodate the
to C9.6 billion until 2032 will be required to strengthen the renewable intermittency and withstand any split event.
German distribution network. Additionally, considering stochastic planning of reserves
Traditionally, system operators have addressed congestion on a daily basis can help improve their effectiveness and
| issues by | investing |     | in new | infrastructure |     | to  | increase the |          |     |             |     |     |     |     |
| --------- | --------- | --- | ------ | -------------- | --- | --- | ------------ | -------- | --- | ----------- | --- | --- | --- | --- |
|           |           |     |        |                |     |     |              | economic |     | efficiency. |     |     |     |     |
network’s initial capacity. However, this approach encounters 2) Introducing fast frequency products: To address the
challenges as peak loads occur only for a few hours each challenge of grid inertia, it is suggested to introduce new
day and are often seasonal. Furthermore, the existing voltage fast-frequency products or virtual inertia products in the
control system at the TSO-DSO interface relies on on-load nearfuture.Learningfromtheimplementationchallenges
tap changers (OLTCs) on the EHV/HV transformers, and from other countries, it is important to establish clear
coordination between TSO and DSO is facilitated through pre-qualification requirements that facilitate easy and
phone communication. The situation becomes more complex profitable participation.
with the growing penetration of renewable energy sources, as 3) Deploying grid forming inverters: Grid forming tech-
manyOLTCsarealreadyoperatingattheirmaximumcapacity nology is commercially available and field-proven for
[52]. Placing a storage asset close to the load center can BESS systems/ hybrid BESS PV systems. It may be
effectivelymeetelectricitydemandduringpeakhourswithout beneficial to explore the option of retrofitting existing
the need for costly upgrades to the incoming transmission BESS plants with grid-forming technology. Furthermore,
or distribution lines, thus providing a significant economic newprojectsshallalreadybemandatedwithgrid-forming
advantage [153] [154]. controls to support future 100% renewable generation-
Accordingto[155],batteries,evenattheircurrentcosts,can based power systems.
potentiallydefertheneedfortransformerreinforcementatthe 4) Enhancing TSO-DSO coordination: Improving the co-
distribution level by up to 15%. According to [156], a cost- ordination scheme between TSOs and DSOs is essential.
benefit of 45% is possible through obviating the distribution The active involvement of DSOs in congestion manage-
grid reinforcement with flexibility measures mainly provided ment through peak shaving services helps alleviate line
by BESS. Work from [157] has presented a detailed cost congestion for TSOs and, most importantly, obviates the
comparison of different reinforcement measures for European need for immediate grid upgrades.

5) Establishing a regional market: Establishing regional [10] Jan Figgener, Christopher Hecht, David Haberschusz, Jakob Bors,
markets is recommended to efficiently procure non- Kai Gerd Spreuer, Kai-Philipp Kairies, Peter Stenzel, and Dirk Uwe
Sauer. The development of battery storage systems in germany: A
frequency ancillary services. Insights can be gained by
marketreview(status2023),2023.
studying the example of Great Britain and China’s local [11] Fraunhofer ISI. Energy Storage in Germany – Present Developments
marketfornon-frequencyancillaryservices.Furthermore, andApplicabilityinChina. Karlsruhe,2020.
[12] Fraktionen SPD, BÜNDNIS 90/DIE GRÜNEN und FDP. Än-
introducing peak shaving products to alleviate grid con-
derungsantrag der fraktionen spd, bÜndnis 90/die grÜnen und fdp
gestion shall also be considered. China’s regional peak- zu dem entwurf eines gesetzes zur anpassung des energiewirtschaft-
shaving markets can be considered an example of build- srechts an unionsrechtliche vorgaben und zur Änderung weit-
erer energierechtlicher vorschriften - drucksachen 20/7310, 20/8165.
ing a market structure.
DeutscherBundestag,Oktober172023.
6) Leveraging BESS resources: Germany can enhance its [13] J.Figgener,B.Tepe,F.Rücker,I.Schoeneberger,C.Hecht,A.Jossen,
grid stability by utilizing existing BESS resources. How- and D. U. Sauer. The influence of frequency containment reserve
flexibilization on the economics of electric vehicle fleet operation.
ever,tofacilitatethis,anewmarketframeworkshouldbe
JournalofEnergyStorage,53:105138,2022.
established.Thisframeworkshouldincentivizeprosumers [14] CameronMurray. Isthegermanutilityscaleenergystoragemarketset
toactivelyparticipateintheflexibilitymarketbyoffering totakeoff?,September2022.
[15] Agora Energiewende. The liberalisation of electricity markets in
home battery storage and EVs. Encouraging a greater
germany:History,developmentandcurrentstatus. https://www.agora-
number of decentralized smaller prosumers can also help energiewende.de/fileadmin/Projekte/2019/\Liberalisation_of_
reduce the risk of a single point of grid failure. Electricity_Markets_in_Germany/\Agora_Energiewende_The_
Liberalisation_of_Electricity_Markets_in_Germany.pdf,2019.
7) Empowering aggregators: Encouraging aggregators in
[16] VB Bordesholm. Batteriespeicher. https://www.vb-bordesholm.de/-
the ancillary market benefits both prosumers and ag- batteriespeicher.html. Accessed:June8,2023.
gregators. It expands prosumer participation, enhancing [17] EnergyStorageNews. 50MWhbatterycompletedinGermanyclaims
Europe’s largest crown. https://www.energy-storage.news/news/
grid flexibility, mitigating single points of failure, and
50mwh-battery-completed-in-germany-claims-europes-largest-crown,
increasing aggregator revenue with more participants. 2023.
It also provides prosumers with a range of aggregator [18] James Ayre. Belectric & vattenfall unveil battery storage facility at
alt daber solar plant. https://cleantechnica.com/2014/12/06/belectric-
optionstochoosefrom,thuscreatingawin-winsituation.
vattenfall-unveil-battery-storage-facility-alt-daber-solar-plant/, 12
8) Flexibility in grid codes: To introduce services such 2014.
as local voltage support, voltage harmonics mitigation, [19] TRACER. Fact Sheet: Innovative electric energy storage "BigBattery
Lausitz". Online,September2019. AccessedonJune8,2023.
oscillation damping, and grid forming capability, it is
[20] Making Batteries a Business: Schwerin Battery Park. Online.
necessary to incorporate flexibility into grid codes. Con-
Availableathttps://energystorage.org/project-profile/making-batteries-
sequently, TSOs and DSOs must consider revising grid a-business-schwerin-battery-park/.
codes at critical bus locations where these advanced [21] BOSCH. A second life for used batteries. Online press release, 10
2016. Availableathttps://www.bosch-press.be/pressportal/be/en/press-
functionalities are needed.
release-562.html.
[22] enercity,Daimlerundenercitymachenersatzteillagerzumenergiespe-
REFERENCES icher. Online, 2 2016. Available at https://www.enercity.de/presse/
pressemeldungen/2016/2016-02-01-batteriespeicher/.
[1] Prognos, Öko-Institut, Wuppertal Institut. Towards a climate-neutral [23] TransnetBW. Netzbooster-Pilotanlage Kupferzell: Projektvorstel-
germany by 2045: How germany can reach its climate targets before lung. https://www.transnetbw.de/files/pdf/netzentwicklung/-projekte/
2050. Executive Summary conducted for Stiftung Klimaneutralität, netzbooster-pilotanlage/broschuere.pdf,2021.
AgoraEnergiewendeandAgoraVerkehrswende,2021. [24] ISEA - Institute for Power Electronics and Electrical Drives.
[2] Arbeitsausgabe der Clearingstelle EEG|KWKG. Gesetz für den aus- M5BAT: Modular Multi-Megawatt Multi-Technology Medium
bau erneuerbarer energien (erneuerbare-energien-gesetz – eeg 2023). Voltage – Battery Storage System. https://www.isea.rwth-
Gesetzesfassungvom4.01.2023,InKraftab1.Februar2023,2023. aachen.de/cms/ISEA/Forschung/Projekte/Oeffentliche-
[3] BrunoBurger.Öffentlichenettostromerzeugungindeutschlandimjahr Projekte/Abgeschlossene-Projekte/pipl/M5BAT/?lidx=1.
2022. https://www.energy-charts.info/?l=en&c=DE,2022. [25] Windfair. German premiere: first “wind + storage” innovation
[4] National Renewable Energy Laboratory (NREL). Grid forming project to be implemented shortly. https://w3.windfair.net/wind-
functional specifications for bpsconnected battery energy storage energy/pr/38649-juwi-mvv-cee-group-wind-farm-turbine-battery-
systems. https://www.nerc.com/comm/RSTC_Reliability_Guidelines/ storage-germany-brandenburg-innovation-tender-pa,September2021.
White_Paper_GFM_Spec_BESS.pdf,June2023. [26] energymatters. 90 mw energy storage project for germany.
[5] European Association for Storage of Energy (EASE). Ancillary https://www.energymatters.com.au/renewable-news/battery-storage-
servicesenergystorageapplicationsforms,August2021. germany-em5190/,November2015.
[6] W.Cole,A.WillFrazier,andC.Augustine.Costprojectionsforutility- [27] Bundesnetzagentur Bundeskartellamt. Monitoring report
scalebatterystorage:2021update. TechnicalReportNREL/TP-6A20- 2022. https://www.bundesnetzagentur.de/SharedDocs/Mediathek/
79236,NationalRenewableEnergyLaboratory,Golden,CO,2021. Monitoringberichte/MonitoringberichtEnergie2022.pdf?__blob=
[7] 50Hertz Transmission GmbH and Amprion GmbH and TenneT TSO publicationFile&v=3.
GmbH and TransnetBW GmbH. Netzentwicklungsplan strom 2037 [28] AgoraEnergiewende.Rap(2015):Reportonthegermanpowersystem.
mit ausblick 2045. Technical report, 50Hertz Transmission GmbH, version 1.2. Study commissioned by agora energiewende, Agora
AmprionGmbH,TenneTTSOGmbH,TransnetBWGmbH,Germany, Energiewende,2015.
2023. [29] Bundeskartellamt. Wettbewerbsverhältnisseimbereichdererzeugung
[8] PrashantPant,HolgerHesse,VivekTejaTanjavooru,MartinCornejo, elektrischerenergie2021marktmachtberichtfebruar2022,2022.
ThomasHamacher,andVedranS.Peric´.Soc-temperaturebasedpower [30] European Parliament and Council of the European Union. Directive
sharingalgorithmforbatteryracksingridancillaryservices. In2023 (eu) 2019/944 of the european parliament and of the council of 5
4thInternationalConferenceonHighVoltageEngineeringandPower june2019oncommonrulesfortheinternalmarketforelectricityand
Systems(ICHVEPS),pages422–427,2023. amendingdirective2012/27/eu.https://eur-lex.europa.eu/legal-content/
[9] JanFiggener,ChristopherHecht,andDirkUweSauer. Batterycharts. EN/TXT/?uri=CELEX:32019L0944,June2019.OfficialJournalofthe
https://www.battery-charts.de/main-page/,2023. EuropeanUnion.

[31] Gesetzüberdieelektrizitäts-undgasversorgung(energiewirtschaftsge- and recommendations for eu renewable power systems. Technical
setz-enwg)§13ekapazitätsreserve.https://www.buzer.de/13e_EnWG. report,TradeRES,2021.
htm. [54] Thomas Wolgast, Stephan Ferenz, and Astrid Nieße. Reactive power
[32] Gesetz über die elektrizitäts- und gasversorgung (energiewirtschafts- markets:Areview. IEEEAccess,10:28397–28410,2022.
gesetz - enwg) § 12h marktgestützte beschaffung nicht frequenzge- [55] Timo Hartmann, Simon Bächle, and Ellen Szczepaniak. Redispatch
bundener systemdienstleistungen. https://www.gesetze-im-internet.de/ 3.0: bottleneck management through decentralised micro-plants?
enwg_2005/__12h.html. https://www.adesso.de/en/news/blog/redispatch-3-0-bottleneck-
[33] Grid reserve / reserve power plant requirement. https: management-through-decentralised-micro-plants-2.jsp,April32023.
//www.bundesnetzagentur.de/EN/Areas/Energy/SecurityOfSupply/ [56] Reactivepower. https://en.kyon-energy.de/glossareintrage/
GridReserve/start.html. blindleistung. AccessedonJuly10,2023.
[34] FritzBraeuer. Techno-economicevaluationofbatterystoragesystems [57] Overview of transmission tariffs in europe: Synthesis 2020.
in industry. PhD thesis, KIT-Fakultät für Wirtschaftswissenschaften, https://eepublicdownloads.entsoe.eu/clean-documents/mcdocuments/
KarlsruherInstitutsfürTechnologie(KIT),2022. Tagdermündlichen l_entsoe_TTO-Report_2020_03.pdf,2020.
Prüfung:October25,2022. [58] ENTSO-E|WGAS. Surveyonancillaryservicesprocurementbalanc-
[35] Bundesnetzagentur für Elektrizität, Gas, Telekommunikation und ingmarketdesign2021,Jun.2022.
Eisenbahnen. BeschlussBK6-17-234,2019. [59] Commissionregulation(eu)2016/1388of17august2016establishing
[36] German Association of Energy and Water Industries (BDEW). Min- a,2016. OfficialJournaloftheEuropeanUnion.
imum Activation Time Period Required for Units with LER in FCR. [60] Bernd Hinzer and Markus Holzapfel. Grid Code Compliance in
PositionPaper,2021. Berlin,9thSeptember2021. Europe:WaystoaFastandSafeGridConnection. July2020.
[37] Tennet. FCR Manual for BSP’s: Requirements and procedures for [61] Commissionregulation(eu)2016/631of14april2016establishinga
supplyofFCR,March2022. ClassificationC1-PublicInformation. networkcodeonrequirementsforgridconnectionofgenerators,2016.
[38] What is frequency containment reserve (fcr)? https://www.next- OfficialJournaloftheEuropeanUnion.
kraftwerke.com/knowledge/frequency-containment-reserve-fcr. [62] EuropeanCommission.CommissionStaffWorkingDocument:Energy
[39] ENTSO-E. Frequencycontainmentreserves(fcr). https://www.entsoe. Storage-UnderpinningadecarbonisedandsecureEUenergysystem.
eu/network_codes/eb/fcr/. SWD2023final,EuropeanCommission,Brussels,March2023.
[40] TenneT GMbh. Prequalification procedure for fcr, afrr, mfrrda and [63] TenneT. Annual market update 2021. https://tennet-drupal.s3.eu-
rod. https://netztransparenz.tennet.eu/fileadmin/user_upload/SO_NL/ central-1.amazonaws.com/default/2022-07/Annual_Market_Update_
20121909_Prekwalificatieproces_ENG.pdf,September2022. 2021_0.pdf,April2022.
[41] Regelleistung.net - Datencenter. Regelleistung.net - Datencenter. [64] Michael Merten. Participation of Battery Storage Systems in the
https://www.regelleistung.net/apps/datacenter/tendering-files/ Automatic Frequency Restoration Reserve Market based on Machine
?productTypes=FCR&markets=CAPACITY&fileTypes=DEMANDS, Learning. PhDthesis,RWTHAachenUniversity,2020.
RESULTS,ANONYMOUS_LIST_OF_BIDS&dateRange=2023- [65] Amprion. MarktgestÜtztebeschaffungvonmomentanreserve. Report,
05,2023-06,2023. Accessed:July3,2023. March2023.
[42] J. Figgener, P. Stenzel, K.-P. Kairies, J. Linßen, D. Haberschusz, and [66] Energiewirtschaftliches Institut an der Universität zu Köln gGmbH,
O.Wessels. Thedevelopmentofstationarybatterystoragesystemsin ef.Ruhr GmbH. Entwicklung der momentanreserve und abschätzung
germany–amarketreview. J.EnergyStorage,29:101153,2020. desbedarfesanfastfrequencyresponseimeuropäischenverbundsys-
[43] ENTSO-E. PICASSO. https://www.entsoe.eu/network_codes/eb/pica- tem.Studie,EnergiewirtschaftlichesInstitutanderUniversitätzuKöln
sso/. Accessed:May21,2023. gGmbH(EWI),Dezember2019. ImAuftragderSiemensAG.
[44] ENTSO-E. Manuallyactivatedreservesinitiative(mari). www.entsoe. [67] EvelinaWiknerandTorbjörnThiringer. Extendingbatterylifetimeby
eu/network_codes/eb/mari/. avoidinghighsoc. AppliedSciences,8:1825,102018.
[45] Konstantinos Oureilidis, Kyriaki-Nefeli Malamaki, Konstantinos Gal- [68] Miguel de la Torre Rodríguez, Marc Scherer, David Whitley, and
los, Achilleas Tsitsimelis, Christos Dikaiakos, Spyros Gkavanoudis, FrankReyer. Frequencycontainmentreservesdimensioningandtarget
MilosCvetkovic,JuanManuelMauricio,JoseMariaMazaOrtega,Jose performanceintheeuropeanpowersystem.In2014IEEEPESGeneral
Luis Martinez Ramos, George Papaioannou, and Charis Demoulias. Meeting|Conference&Exposition,pages1–5,2014.
Ancillaryservicesmarketdesignindistributionnetworks:Reviewand [69] ICSInvestigationExpertPanel.ContinentalEuropeSynchronousArea
identificationofbarriers. Energies,13(4):917,Feb2020. Separation on 08 January 2021—Final Report. Technical report, The
[46] ENTSO-E. Trans european replacement reserves exchange (terre). ExpertPanelontheSeparationoftheContinentalEuropeSynchronous
https://www.entsoe.eu/network_codes/eb/terre/. Areaof08January2021,2021.
[47] ENTSO-E. Balancing Report. https://ee-public-nc-downloads. [70] ENTSO-E. Operational Reserve ad hoc Team Report.
azureedge.net/strapi-test-assets/strapi-assets/2022_ENTSO_E_ https://eepublicdownloads.azureedge.net/clean-documents/
Balancing_Report_Web_2bddb9ad4f.pdf,2022. Publications/Market%20Committee%20publications/ENTSO-
[48] TERRE Trans European Replacement Reserves Exchange RRIF & E_Balancing_Report_2020.pdf,2012.
TERREStakeholdersWorkshop,March192018. Workshop. [71] ENTSO-E. High penetration of power electronic interfaced power
[49] Updateonbalancingplatforms. URL:https://eepublicdownloads.blob. sources and the potential contribution of grid forming converters.
core.windows.net/public-cdn-container/clean-documents/Network% Technicalreport,Belgium,2020.
20codes%20documents/MESC/2023%20MESC%20documents/ [72] J. Ungerland and P. Pant. Challenges of modelling equivalent active
230309_MESC_TOP_5.1_Update_on_balancing_platforms.pdf,March distribution grids under consideration of grid forming inverters. In
2023. [Online]. Virtual19thWindIntegrationWorkshop,2020.
[50] European Union Electricity Market Glossary. Non-frequency ancil- [73] Christian Sperling. Who is disrupting the utility frequency?
lary services. https://emissions-euets.com/internal-electricity-market- https://www.next-kraftwerke.com/energy-blog/who-is-disrupting-the-
glossary/1821-non-frequency-related-ancillary-services. Published:16 utility-frequency,February152019.
June2017,LastUpdated:24April2023. [74] 50 Hertz Transmission GmbH and Co KG and Amprion
[51] G. Blumberg, C. Wagner, W. Lehnert, M. Bucksteeg, and M. Greve. GmbH and TenneT TSO GmbH and TransnetBW GmbH.
Marktgestützte beschaffung von blindleistung: Bericht im vorhaben 4-tso paper on requirements for grid-forming converters.
„sdl-zukunft“. Technicalreport,RuhrGmbH,2021. https://www.netztransparenz.de/portals/1/220504_-_4-TSO_Paper_
[52] Kseniia Sinitsyna, Maik Staudt, René Sturm, Carmen Calpe, Sebas- on_Requirements_for_Grid-Forming_Converters.pdf,2020.
tian Wende von Berg, Katharina Brauns, Lothar Löwer, Benjamin [75] H. Popella, T. Hennig, M. Kaiser, J. Massmann, L. Müller, and
Requardt,ArneWessel,ZhenqiWang,FábioRetorta,JoãoVieiraSilva, R. Pfeiffer. Necessary development of inverter-based generation with
and Bernardo Silva. The german demonstration: Flexibility of active gridformingcapabilitiesingermany. In20thInternationalWorkshop
andreactivepowerfromhvdistributiongridtoehvtransmissiongrid. onLarge-ScaleIntegrationofWindPowerintoPowerSystemsaswell
projectreport,MITNETZSTROM,E.ON,FraunhoferIEE/Uni-Kassel as on Transmission Networks for Offshore Wind Power Plants (WIW
"e2n",INESCTEC,Feb.2022. EU-SYSFLEX. 2021),volume2021,pages125–129,2021.
[53] JosSijm,RicardoFaia,DeboraSaoJose,BodeWildt,andVriesLau- [76] Project MIGRATE. Report on systemic issues. Report, Project
rensde. Designofancillaryservicemarketsandproducts:Challenges MIGRATE,December152016.

[77] 50Hertz Transmission GmbH and Amprion GmbH and Tennet TSO operating conditions. In IEEE PES ISGT Europe 2024 - Innovative
GmbH and TransnetBW GmbH. Bewertung der Systemstabilität, SmartGridTechnologies,Dubrovnik,Croatia,October2024.
BegleitdokumentzumNetzentwicklungsplanStrom2030,Version2019, [100] Njegos Jankovic, Javier Roldan-Perez, Milan Prodanovic, Jon Are
zweiterEntwurf.50HertzTransmissionGmbH,Berlin,Germany,2019. Suul, Salvatore D’Arco, and Luis Rouco Rodriguez. Power oscilla-
[78] Semich Impram, Secil Varbak Nese, and Bülent Oral. Challenges of tion damping method suitable for network reconfigurations based on
renewable energy penetration on power system flexibility: A survey. converterinterfacedgenerationandcombineduseofactiveandreactive
EnergyStrategyReviews,31:100539,2020. powers. InternationalJournalofElectricalPower&EnergySystems,
[79] AMPRION. Amprion Market Report 2021: Current Trends 149:109010,2023.
and Developments of the Market and Grid Situation. [101] Mohammad Amin and Marta Molinas. Understanding the origin of
https://www.amprion.net/Dokumente/Strommarkt/Marktbericht/ oscillatoryphenomenaobservedbetweenwindfarmsandhvdcsystems.
Amprion_Market_Report_2021.pdf,2021. IEEEJournalofEmergingandSelectedTopicsinPowerElectronics,
[80] ENTSO-E (European Network of Transmission System Operators for 5(1):378–392,2017.
Electricity). Fast frequency reserve - solution to the nordic inertia [102] WeishengWang,GuanghuiLi,andJianboGuo.Large-scalerenewable
challenge. December132019. energytransmissionbyhvdc:Challengesandproposals. Engineering,
[81] North American Electric Reliability Corporation (NERC) Inverter- 19:252–267,2022.
Based Resource Performance Task Force (IRPTF). Fast frequency [103] Yunzhi Cheng, Lingling Fan, Jonathan Rose, Shun-Hsien Huang,
response concepts and bulk power system reliability needs. White John Schmall, Xiaoyu Wang, Xiaorong Xie, Jan Shair, Jayanth R.
paper,NorthAmericanElectricReliabilityCorporation,March2020. Ramamurthy,NileshModi,ChunLi,ChenWang,ShahilShah,Bikash
[82] Entose. The inertia challenge in europe – present and long-term Pal,ZhixinMiao,AndrewIsaacs,JeanMahseredjian,andJennyZhou.
perspective. Technical report, Insight Report, August 2021. Final Real-worldsubsynchronousoscillationeventsinpowergridswithhigh
versionafterACERopinion. penetrationsofinverter-basedresources. IEEETransactionsonPower
[83] ENTSO-E. Nordic balancing philosophy. Technical report, 2021. Systems,38(1):316–330,2023.
Updated2021-11-10. [104] MIGRATE. Public deliverable D1.1: Current and arising issues
caused by increasing power electronics penetration - Synthesis and
[84] Indira Alcaide-Godinez, Feifei Bai, Tapan Kumar Saha, and Rizah
questionnaire-,2018. MIGRATE.
Memisevic.Contingencyreserveevaluationforfastfrequencyresponse
ofmultiplebatteryenergystoragesystemsinalarge-scalepowergrid. [105] Western Electricity Coordinating Council Modeling and Validation
CSEEJournalofPowerandEnergySystems,pages1–10,2022. Work Group. WECC Battery Storage Dynamic Modeling Guideline.
Technical report, WECC Renewable Energy Modeling Task Force,
[85] AustralianEnergyMarketOperator(AEMO). Fastfrequencyresponse
November2016.
in the nem. Working Paper, 2017. Future Power System Security
[106] Prashant Pant, Vedran S. Peric´, and Thomas Hamacher. System
Program.
identificationbasedadaptivedroopcontrolstrategyforpredominantly
[86] National Grid ESO. New dynamic services (dc/dm/dr).
resistivemicrogrids. In2023IEEEPower&EnergySocietyGeneral
https://www.nationalgrideso.com/industry-information/balancing-
Meeting(PESGM),pages1–5,2023.
services/frequency-response-services/new-dynamic-services-dcdmdr.
[107] Prashant Pant, Thomas Hamacher, Federico Martin Ibanez, and Ve-
[87] IEEE.Ieeestandardforinterconnectionandinteroperabilityofinverter-
dranS.Peric´.Systemidentificationbasedgridagnosticadaptivedroop
based resources (ibrs) interconnecting with associated transmission
control strategy. In 2023 IEEE Belgrade PowerTech, pages 01–06,
electricpowersystems. IEEEStd2800-2022,pages1–180,2022.
2023.
[88] U.Tamrakar,D.Shrestha,M.Mahajan,andT.Hansen.Virtualinertia:
[108] Gurupraanesh Raman, Jimmy Chih-Hsien Peng, and Hatem H.
Currenttrendsandfuturedirections. MDPI,2017.
Zeineldin. Optimal damping recovery scheme for droop-controlled
[89] B. Poola. System Normal Approaches for Power System Stability
inverter-based microgrids. IEEE Transactions on Smart Grid,
AnalysisandControl. Doctoraldissertation,ETHZurich,2019.
11(4):2805–2815,2020.
[90] Amprion. Amprion Market Report2022. https://www.amprion.net/
[109] Vahan Gevorgian, Przemyslaw Koralewicz, Shahil Shah, Emanuel
Dokumente/Strommarkt/Marktbericht/2022/AMP_Market_Report_
Mendiola, Robb Wallen, and Hugo Villegas Pico. Photovoltaic plant
2022.pdf,2022.
andbatteryenergystoragesystemintegrationatnrel’sflatironscampus.
[91] AEMC. Efficient provision of inertia - Consultation pa- TechnicalReportNREL/TP-5D00-81104,NationalRenewableEnergy
per. https://www.aemc.gov.au/sites/default/files/2023-03/ERC0339% Laboratory,Golden,CO,2022.
20-%20Consultation%20Paper.pdf,2ndMarch2023.
[110] C.Duggan,X.Liu,P.Brogan,R.Best,andD.J.Morrow. Verylow-
[92] Luis Badesa, Goran Strbac, Matt Magill, and Biljana Stojkovska. frequency oscillation source localization on ireland’s power system.
Ancillary services in great britain during the covid-19 lockdown: A IEEEOpenJournalofIndustryApplications,3:192–201,2022.
glimpseofthecarbon-freefuture. AppliedEnergy,285:116500,2021. [111] RebeccaStaudenmaier. Germany’sscholzpledgesrapidonshorewind
[93] Jonathan Spencer Jones. National grid eso advances inertia power expansion. https://www.dw.com/en/germanys-scholz-pledges-
management on gb grid. https://www.smart-energy.com/industry- rapid-onshore-wind-power-expansion/a-64616000,February52023.
sectors/energy-grid-management/national-grid-eso-advances-inertia- [112] Australian Energy Market Operator (AEMO). Market Ancillary
management-on-gb-grid/,Apr2022. Service Specification Consultation. https://aemo.com.au/-
[94] NationalGridESO. Workgroupconsultationgc0137:Minimumspec- /media/files/stakeholder_consultation/consultations/nem-consultations/
ification required for provision of gb grid forming (gbgf) capability 2022/amendment-of-the-mass/mass-issues-paper.pdf?la=en, May
(formerly virtual synchronous machine/vsm capability). https://www. 2022.
nationalgrideso.com/document/189381/download,March2021. [113] Manuel Garcia and Ross Baldick. Requirements for interdependent
[95] Svenskakraftnät(SVK).Explanatorydocumentfortheamendednordic reservetypesprovidingprimaryfrequencycontrol. IEEETransactions
synchronousareamethodologyforadditionalpropertiesoffcr.Online, onPowerSystems,37(1):51–64,2022.
February2023. [114] Australian Energy Market Commission (AEMC). Fast frequency
[96] Zeenat Hameed, Chresten Træholt, and Seyedmostafa Hashemi. In- responsemarketancillaryservice. Finalreport,July2021.
vestigating the participation of battery energy storage systems in the [115] Fingrid. Fast Frequency Reserve (FFR). https://www.fingrid.fi/en/
nordicancillaryservicesmarketsfromabusinessperspective. Journal electricity-market/reserves_and_balancing/fast-frequency-reserve/#
ofEnergyStorage,58:106464,2023. technical-requirements,2023.
[97] Lingling Fan and Zhixin Miao. Wind in weak grids: 4 hz or 30 hz [116] IRENA. Innovation landscape brief: Innovative ancillary services.
oscillations? IEEETransactionsonPowerSystems,33(5):5803–5804, Technicalreport,InternationalRenewableEnergyAgency,AbuDhabi,
2018. 2019.
[98] YasserAbdel-RadyIbrahimMohamedandEhabF.El-Saadany. Adap- [117] K.Balasubramanian. Germany’sduckcurve–Integratingrenewables
tivedecentralizeddroopcontrollertopreservepowersharingstability into smart grids. https://www.gridx.ai/blog/germanys-duck-curve-
of paralleled inverters in distributed generation microgrids. IEEE integrating-renewables-into-smart-grids. Published:June5,2023.
TransactionsonPowerElectronics,23(6):2806–2816,2008. [118] EirGrid. Ancillary services report 2021/2022. https:
[99] MartaGomis-Domènech,Thai-PhuongDo,andRaphaëlCaire.Perfor- //www.eirgridgroup.com/site-files/library/EirGrid/AS-OSC-
manceofIBRPowerOscillationsDamperaccordingtopowersystem Report_2021-22(1).pdf,August2022.

[119] CarmenCardozo,YannickVernay,GuillaumeDenis,ThibaultPrevost, [140] BenjaminAustandAndreasHorsch. Negativemarketpricesonpower
MarkelZubiaga,andJuanValera. Osmosegrid-formingperformance exchanges: Evidence and policy implications from germany. The
assessment within multiservice storage connected to the transmission ElectricityJournal,33(3),2020.
grid. 082020. [141] Edgar Meza. Germany’s grid expansion going slower than planned.
[120] JuliaMatevosyan. Capturingthebenefitsofgridformingbatteries:A https://www.cleanenergywire.org/news/germanys-grid-expansion-
going-slower-planned,October2021.
uniquewindowofopportunity.https://www.esig.energy/capturing\-the-
benefits-of-grid-forming-batteries-a-unique-window-of\-opportunity/, [142] KseniaPoplavskaya,MarineJoos,VincentKrakowski,KasparKnorr,
2023. andLaurensdeVries. Redispatchandbalancing:Samebutdifferent.
[121] NationalGridESO.Stabilitypathfinder,draftgridcode–gridforming links, conflicts and solutions. In 2020 17th International Conference
andrequestforinformationfeedback. Website,2019. Availableonline ontheEuropeanEnergyMarket(EEM),pages1–6,2020.
|     |     |     |     |     |     |     |     | [143] GermanEnergyIndustryAct(Energiewirtschaftsgesetz–EnWG).http: |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
atwww.nationalgrideso.com/document/185306/download.
//www.gesetze-im-internet.de/enwg_2005/EnWG.pdf.
| [122] ENTSO-E. |     | Fault current | contribution |     | from | PPMS | & HVDC: |     |     |     |     |     |     |     |     |
| -------------- | --- | ------------- | ------------ | --- | ---- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
ENTSO-E guidance document for national implementation [144] Eric Heymann. Costs of electricity generation: System
for network codes on grid connection: Draft of expert group costs matter. https://www.dbresearch.de/PROD/RPS_DE-
FFCI. ENTSO-E, Draft Document, March 29 2017. Available PROD/PROD0000000000528292/Costs_of_electricity_generation%
3A_System_costs_matt.PDF,May2023.
| online: |     | https://eepublicdownloads.entsoe.eu/clean-documents/ |     |     |     |     |     |           |         |         |     |       |             |        |             |
| ------- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --------- | ------- | ------- | --- | ----- | ----------- | ------ | ----------- |
|         |     |                                                      |     |     |     |     |     | [145] How | battery | storage | can | bring | flexibility | to the | power grid. |
Network%20codes%20documents/Implementation/CNC/IGD-
Fault_Current_Contribution.pdf. https://en.kyon-energy.de/blog/wie-batteriespeicher-flexibilitat-ins-
[123] InmaculadaMartínezSanz,BiljanaStojkovska,AngelaWilks,Jonathan stromnetz-bringen-konnen.
Horne,AliR.Ahmadi,andTatianaUstinova. Enhancingtransmission [146] KerstineAppunn. New"powergridactionplan"toacceleratenetwork
|                   |              |                                             |              |     |         |       |             | development. |            | CleanEnergyWire,August2018. |                                                |     |     |     |     |
| ----------------- | ------------ | ------------------------------------------- | ------------ | --- | ------- | ----- | ----------- | ------------ | ---------- | --------------------------- | ---------------------------------------------- | --- | --- | --- | --- |
| and               | distribution | system                                      | coordination | and | control | in gb | using power |              |            |                             |                                                |     |     |     |     |
|                   |              |                                             |              |     |         |       |             | [147] BMWi.  | Redispatch |                             | 3.0. https://www.bmwk.de/Redaktion/DE/Artikel/ |     |     |     |     |
| servicesfromders. |              | TheJournalofEngineering,2019(18):4911–4915, |              |     |         |       |             |              |            |                             |                                                |     |     |     |     |
2019. Digitale-Welt/GAIA-X-Use-Cases/redispatch-30.html,2023.
[124] EIRGRID and SONI. DS3 System Services: Portfolio Capability [148] Study on redispatch 3.0 presented. https://www.transnetbw.de/de/
Analysis. Technicalreport,November2014. newsroom/presseinformationen/studie-zu-redispatch-3-0-vorgestellt.
|              |         |        |             |       |             |     |           | [149] Clémentine |                | Straub, | Jean Maeght, | Camille    | Pache, | Patrick | Panciatici,     |
| ------------ | ------- | ------ | ----------- | ----- | ----------- | --- | --------- | ---------------- | -------------- | ------- | ------------ | ---------- | ------ | ------- | --------------- |
| [125] Atheer | Habash, | Grazia | Todeschini, | Jacob | W.J. Lynch, | and | Zia Emin. |                  |                |         |              |            |        |         |                 |
|              |         |        |             |       |             |     |           | and              | Ram Rajagopal. |         | Congestion   | management |        | within  | a multi-service |
Harmonicmitigationasancillaryserviceprovidedbymultiplephoto-
|                   |     |                                           |     |     |     |     |     | scheduling |     | coordination | scheme | for | large battery | storage | systems. In |
| ----------------- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- | ------------ | ------ | --- | ------------- | ------- | ----------- |
| voltaicinverters. |     | In2021IEEEPESInnovativeSmartGridTechnolo- |     |     |     |     |     |            |     |              |        |     |               |         |             |
giesEurope(ISGTEurope),pages1–5,2021. 2019IEEEMilanPowerTech,pages1–6,2019.
[126] EASY-RES.EnableAncillaryServicesbyRESD5.1ReportReviewing [150] GesetzüberdieElektrizitäts-undGasversorgung(Energiewirtschafts-
|                                      |     |     |     |     |                       |     |     | gesetz | - EnWG) | §   | 11b Ausnahme |     | für Energiespeicheranlagen, |     | Fes- |
| ------------------------------------ | --- | --- | --- | --- | --------------------- | --- | --- | ------ | ------- | --- | ------------ | --- | --------------------------- | --- | ---- |
| theCurrentMarketRegulatoryFramework. |     |     |     |     | DeliverableD5.1,EASY- |     |     |        |         |     |              |     |                             |     |      |
tlegungskompetenz.https://www.gesetze-im-internet.de/enwg_2005/__
| RES | Consortium. | Project | funded | by the | European | Union’s | Horizon |     |     |     |     |     |     |     |     |
| --- | ----------- | ------- | ------ | ------ | -------- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
11b.html.
2020ResearchandInnovationProgramme.
|     |     |     |     |     |     |     |     | [151] International |     | Energy | Agency | (IEA). | Unlocking | the | Potential of |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ------ | ------ | ------ | --------- | --- | ------------ |
[127] dena - Deutsche Energie-Agentur GmbH. Ancillary Services Study Distributed Energy Resources: Power System Opportunities and Best
2030.Securityandreliabilityofapowersupplywithahighpercentage
|              |     |               |         |           |         |      |            | Practices.                                                         | 2022. |     |     |     |     |     |     |
| ------------ | --- | ------------- | ------- | --------- | ------- | ---- | ---------- | ------------------------------------------------------------------ | ----- | --- | --- | --- | --- | --- | --- |
| of renewable |     | energy. Final | report. | Technical | report, | dena | - Deutsche |                                                                    |       |     |     |     |     |     |     |
|              |     |               |         |           |         |      |            | [152] J.Büchner,J.Katzfey,O.Flörcken,A.Moser,H.Schuster,S.Dierkes, |       |     |     |     |     |     |     |
Energie-AgenturGmbH,2023.
|                             |     |                                            |     |     |     |     |     | T.           | van Leeuwen, | L.  | Verheggen,       | M.  | Uslar, and           | M. van | Amelsvoort. |
| --------------------------- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | ------------ | ------------ | --- | ---------------- | --- | -------------------- | ------ | ----------- |
| [128] Blackstartcapability. |     | https://en.kyon-energy.de/glossareintrage/ |     |     |     |     |     |              |              |     |                  |     |                      |        |             |
|                             |     |                                            |     |     |     |     |     | Moderne      | verteilnetze |     | für deutschland. |     | verteilernetzstudie. |        | Available   |
| schwarzstartfahigkeit.      |     |                                            |     |     |     |     |     | online,2014. |              |     |                  |     |                      |        |             |
[129] Nitish Saraf, Kenneth McIntyre, John Dumas, and Surya Santoso. [153] DanielKucevic,RebeccaMeißner,AndreasJossen,andHolgerHesse.
| The | annual | black start | service | selection | analysis | of ercot | grid. IEEE |         |        |         |         |       |             |                 |      |
| --- | ------ | ----------- | ------- | --------- | -------- | -------- | ---------- | ------- | ------ | ------- | ------- | ----- | ----------- | --------------- | ---- |
|     |        |             |         |           |          |          |            | Battery | energy | storage | systems | as an | alternative | to conventional | grid |
TransactionsonPowerSystems,24(4):1867–1874,2009.
|           |             |        |            |     |          |         |         | reinforcement. |     | InProceedingsoftheInternationalConferenceonAp- |     |     |     |     |     |
| --------- | ----------- | ------ | ---------- | --- | -------- | ------- | ------- | -------------- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- |
| [130] EDF | Renewables. | Energy | transition | in  | germany: | Battery | storage | as             |     |                                                |     |     |     |     |     |
pliedEnergy,Bangkok,Thailand(Virtual),November29-December
key technology in the energy transition. https://www.edf-re.de/assets/ 22021. PaperID:0048.
media/Main/PDFs/201124_EDF-Whitepaper-EN.pdf,2020. [154] StéphaneAllard,VincentDebusschere,SilvanaMima,TuanTranQuoc,
[131] HydrogenEurope. Publicconsultation:Revisionoftheeu’selectricity Nouredine Hadjsaid, and et al. Considering distribution grids and
| marketdesign. |     | Online,2023. | Date:12/02/202322:13:08. |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------- | --- | ------------ | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
localflexibilitiesintheprospectivedevelopmentoftheeuropeanpower
| [132] Smart | Energy | Europe. | Unlocking | demand-side |     | flexibility | through | a             |     |                             |     |     |     |     |     |
| ----------- | ------ | ------- | --------- | ----------- | --- | ----------- | ------- | ------------- | --- | --------------------------- | --- | --- | --- | --- | --- |
|             |        |         |           |             |     |             |         | systemby2050. |     | AppliedEnergy,270,July2020. |     |     |     |     |     |
consumer-centricrevisionoftheeuelectricitymarketdesign. Position [155] Ruchi Gupta, Alejandro Pena-Bello, Kai Nino Streicher, Cattia Ro-
paperandsuggestionforamendments,April2023. duner,YamshidFarhat,DavidThöni,MartinKumarPatel,andDavid
[133] TransnetBW. Mains booster copper cell. https://www.transnetbw.de/ Parra. Spatial analysis of distribution grid capacity and costs to
de/netzentwicklung/projekte/netzbooster-kupferzell. enablemassivedeploymentofpv,electricmobilityandelectricheating.
[134] Zhang,Gao,Fan,Menghua,andlv,Chen. Mechanismdesignofchina AppliedEnergy,287:116504,2021.
ancillary service market considering provincial and inter-provincial [156] MatthiasResch,JochenBühler,BirgitSchachler,andAndreasSumper.
marketcharacteristics. SHSWebConf.,163:02037,2023. Techno-economicassessmentofflexibilityoptionsversusgridexpan-
[135] Wärtsilä. Can ancillary services markets shape china’s new energy sion in distribution grids. IEEE Transactions on Power Systems,
landscape? https://www.wartsila.cn/en/energy/powersystem/ancillary- 36(5):3830–3839,2021.
services-article,July62022. [157] Hussein Abdeltawab and Yasser Abdel-Rady I. Mohamed. Energy
|     |     |     |     |     |     |     |     | storage | planning | for | profitability | maximization |     | by power | trading and |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | --- | ------------- | ------------ | --- | -------- | ----------- |
[136] ChenghuiTangandHaoyuanQu.Developmentofancillaryservicesin
theelectricitymarketofchinaanditsjointmarketoptimizationmodel. ancillary services participation. IEEE Systems Journal, 16(2):1909–
| IOPConferenceSeries:EarthandEnvironmentalScience,571:012029, |     |     |     |     |     |     |     | 1920,2022. |     |     |     |     |     |     |     |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
112020.
| [137] StefanEnglberger,AndreasJossen,andHolgerHesse. |           |                                               |           |                                |                | Unlockingthe |              |     |     |     |     |     |     |     |     |
| ---------------------------------------------------- | --------- | --------------------------------------------- | --------- | ------------------------------ | -------------- | ------------ | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
| potential                                            | of        | battery storage                               | with      | the dynamic                    |                | stacking     | of multiple  |     |     |     |     |     |     |     |     |
| applications.                                        |           | CellReportsPhysicalScience,1(11):100238,2020. |           |                                |                |              |              |     |     |     |     |     |     |     |     |
| [138] Elian                                          | Pusceddu, | Behnam                                        | Zakeri,   | and                            | Giorgio        | Castagneto   | Gissey.      |     |     |     |     |     |     |     |     |
| Synergies                                            | between   | energy                                        | arbitrage | and                            | fast frequency |              | response for |     |     |     |     |     |     |     |     |
| batteryenergystoragesystems.                         |           |                                               |           | AppliedEnergy,283:116274,2021. |                |              |              |     |     |     |     |     |     |     |     |
| [139] Cameron                                        | Murray.   | German                                        | battery   | storage                        | developer      |              | kyon energy: |     |     |     |     |     |     |     |     |
| grid                                                 | frequency | service                                       | prices    | to drop                        | but            | ‘remain      | elevated’.   |     |     |     |     |     |     |     |     |
https://www.energy-storage.news/german-battery-storage-developer-
kyon-energy-grid-frequency-service-prices-to-drop-but-remain-
elevated/,October2022.