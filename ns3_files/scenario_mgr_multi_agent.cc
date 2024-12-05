#include <chrono>
#include <filesystem>
#include <map>
#include <string>
#include <iostream>
#include <sstream>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/ssid.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ns3-ai-module.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-mac.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("scenario");

/*** ns3-ai structures definitions ***/

#define DEFAULT_MEMBLOCK_KEY 2333

struct sEnv
{
  double fairness;
  double latency;
  double plr;
  double time;
  double tx_list[10];
  double lost_list[10];
  double throughput[10];
  double collisions[10];
} Packed;

struct sAct
{
  bool end_warmup;
  int cw[10];
} Packed;

Ns3AIRL<sEnv, sAct> * m_env = new Ns3AIRL<sEnv, sAct> (DEFAULT_MEMBLOCK_KEY);

/***** Functions declarations *****/
void splitString(std::string& input, char delimiter,
                 std::string arr[], int& index)
{
    std::istringstream stream(input);
    std::string token;
    while (getline(stream, token, delimiter)) {
        arr[index++] = token;
    }
}

double global_collinsions_ap = 0;
double global_drop_list[10];
double previous_global_drop_list[10];
void ResetMonitor ();
void InstallTrafficGenerator (Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, uint32_t port,
                              DataRate offeredLoad, uint32_t packetSize);
void PopulateARPcache ();
void ExecuteAction (std::string agentName, double dataRate, double distance, uint32_t nWifi, int cheaterNumber);
void SetNetworkConfiguration (int cw_idx);
void SetNetworkConfigurationCheater (int cw_idx, int cheaterNum);
void TxDrop (std::string context,uint8_t reason, Ptr< const WifiMpdu > mpdu, const WifiTxVector &txVector)
{ 
  char delimiter = '/';
  std::string arrayOfSubStr[100];
  int index = 0;
  splitString(context, delimiter, arrayOfSubStr, index);
  int collisionIndex = stoi(arrayOfSubStr[2]);
  global_drop_list[collisionIndex]++;
  std::cout << "Droped phy packet " << Simulator::Now().GetSeconds() << std::endl;
}

void TxDropMac (std::string context, const Ptr< const Packet > packet)
{ 
  char delimiter = '/';
  std::string arrayOfSubStr[100];
  int index = 0;
  splitString(context, delimiter, arrayOfSubStr, index);
  int collisionIndex = stoi(arrayOfSubStr[2]);
  global_drop_list[collisionIndex]++;
  std::cout << "Droped phy packet " << std::endl;
}

void MonitorRetransmissions(std::string context, Ptr<const Packet> packet) {
    WifiMacHeader header;
    std::string arrayOfSubStr[100];
    char delimiter = '/';
    int index = 0;
    
    if (packet->PeekHeader(header)) { // Wyodrębnienie nagłówka WifiMacHeader
      int IsRetry = header.IsRetry();
        if (IsRetry == 1) {
            splitString(context, delimiter, arrayOfSubStr, index);
            int collisionIndex = stoi(arrayOfSubStr[2]);
            if (collisionIndex == 0){
              global_collinsions_ap++;
            } else {
              global_drop_list[collisionIndex-1]++;
            }
        }
    }
}
void IpCheck(std::string context, Ptr<const Packet> packet) {
    WifiMacHeader header;
    std::string arrayOfSubStr[100];
    char delimiter = '/';
    int index = 0;

    if (packet->PeekHeader(header)) { // Wyodrębnienie nagłówka WifiMacHeader
      int IsRetry = header.IsRetry();
        if (IsRetry == 1) {
            splitString(context, delimiter, arrayOfSubStr, index);
            int collisionIndex = stoi(arrayOfSubStr[2]);
            if (collisionIndex == 0){
              
              global_collinsions_ap++;
            } else {
              global_drop_list[collisionIndex-1]++;
            }
        }
    }
}

/***** Global variables and constants *****/

double fuzzTime = 5.;
double simulationTime = 5.;
double interactionTime = 0.5;
double warmupEndTime = 0.;
bool simulationPhase = false;
bool useMabAgent = false;



double previousRX = 0;
double previousTX = 0;
double previousLost = 0;
Time previousDelay = Seconds(0);

Ptr<FlowMonitor> monitor;
std::map<FlowId, FlowMonitor::FlowStats> previousStats;

std::ostringstream csvLogOutput;

/***** Main with scenario definition *****/

int
main (int argc, char *argv[])
{
  // Initialize default simulation parameters
  uint32_t nWifi = 10;
  uint32_t maxQueueSize = 100;
  uint32_t packetSize = 1500;
  uint32_t dataRate = 110;
  uint32_t channelWidth = 20;
  int cheaterNumber = 1;
  double distance = 10.;

  std::string agentName = "wifi";
  std::string pcapName = "Analiza.pcap";
  std::string csvPath = "results.csv";
  std::string csvLogPath = "logs.csv";
  std::string flowmonPath = "flowmon.xml";

  int cw_idx = -1;
  bool rts_cts = false;
  bool ampdu = true;

  // Parse command line arguments
  CommandLine cmd;
  cmd.AddValue ("agentName", "Name of the agent", agentName);
  cmd.AddValue ("ampdu", "Enable A-MPDU (only for wifi agent)", ampdu);
  cmd.AddValue ("channelWidth", "Channel width (MHz)", channelWidth);
  cmd.AddValue ("csvLogPath", "Path to output CSV log file", csvLogPath);
  cmd.AddValue ("csvPath", "Path to output CSV file", csvPath);
  cmd.AddValue ("cw", "Contention window (const CW = 2 ^ (4 + x) if x >= 0) (only for wifi agent)", cw_idx);
  cmd.AddValue ("dataRate", "Traffic generator data rate (Mb/s)", dataRate);
  cmd.AddValue ("distance", "Max distance between AP and STAs (m)", distance);
  cmd.AddValue ("flowmonPath", "Path to output flow monitor XML file", flowmonPath);
  cmd.AddValue ("fuzzTime", "Maximum fuzz value (s)", fuzzTime);
  cmd.AddValue ("interactionTime", "Time between agent actions (s)", interactionTime);
  cmd.AddValue ("maxQueueSize", "Max queue size (packets)", maxQueueSize);
  cmd.AddValue ("nWifi", "Number of stations", nWifi);
  cmd.AddValue ("packetSize", "Packets size (B)", packetSize);
  cmd.AddValue ("pcapName", "Name of a PCAP file generated from the AP", pcapName);
  cmd.AddValue ("rtsCts", "Enable RTS/CTS (only for wifi agent)", rts_cts);
  cmd.AddValue ("simulationTime", "Duration of simulation (s)", simulationTime);
  cmd.AddValue ("cheaterNumber", "Number of cheaters in network", cheaterNumber);
  cmd.Parse (argc, argv);

  // Print simulation settings to screen
  std::cout << std::endl
            << "Simulating an IEEE 802.11ax devices with the following settings:" << std::endl
            << "- agent: " << agentName << std::endl
            << "- frequency band: 5 GHz" << std::endl
            << "- max data rate: " << dataRate << " Mb/s" << std::endl
            << "- channel width: " << channelWidth << " Mhz" << std::endl
            << "- packets size: " << packetSize << " B" << std::endl
            << "- max queue size: " << maxQueueSize << " packets" << std::endl
            << "- number of stations: " << nWifi << std::endl
            << "- max distance between AP and STAs: " << distance << " m" << std::endl
            << "- simulation time: " << simulationTime << " s" << std::endl
            << "- max fuzz time: " << fuzzTime << " s" << std::endl
            << "- interaction time: " << interactionTime << " s" << std::endl;

  if (agentName == "wifi")
    {
      std::cout << "- CW: " << (cw_idx >= 0 ? "2 ^ (4 + " + std::to_string (cw_idx) + ")" : "default" ) << std::endl
                << "- RTS/CTS: " << (rts_cts ? "enabled" : "disabled") << std::endl
                << "- A-MPDU: " << (ampdu ? "enabled" : "disabled") << std::endl;
    }

  useMabAgent = agentName != "wifi";

  // Create AP and stations
  NodeContainer wifiApNode (1);
  NodeContainer wifiStaNodes (nWifi);

  // Configure mobility model
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::UniformDiscPositionAllocator",
                                 "X", DoubleValue (0.0),
                                 "Y", DoubleValue (0.0),
                                 "rho", DoubleValue (distance));

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNodes);

  // Make sure AP is at the center
  Ptr<MobilityModel> apMobility = wifiApNode.Get (0)->GetObject<MobilityModel> ();
  apMobility->SetPosition (Vector (0.0, 0.0, 0.0));

  // Print position of each node
  std::cout << std::endl << "Node positions:" << std::endl;

  // AP position
  Ptr<MobilityModel> position = wifiApNode.Get (0)->GetObject<MobilityModel> ();
  Vector pos = position->GetPosition ();
  std::cout << "AP:\tx=" << pos.x << ", y=" << pos.y << std::endl;

  // Stations positions
  for (auto node = wifiStaNodes.Begin (); node != wifiStaNodes.End (); ++node)
    {
      position = (*node)->GetObject<MobilityModel> ();
      pos = position->GetPosition ();
      std::cout << "Sta " << (*node)->GetId () << ":\tx=" << pos.x << ", y=" << pos.y << std::endl;
    }

  std::cout << std::endl;

  // Configure wireless channel
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channelHelper = YansWifiChannelHelper::Default ();
  phy.SetChannel (channelHelper.Create ());

  // Configure MAC layer
  
  WifiHelper wifi;
  WifiMacHelper mac;
  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  mac.SetType("ns3::AdhocWifiMac");

  // Set SSID
  Ssid ssid = Ssid ("ns3-80211ax");
  // mac.SetType ("ns3::StaWifiMac",
  //              "Ssid", SsidValue (ssid),
  //              "MaxMissedBeacons", UintegerValue (1000)); // prevents exhaustion of association IDs

  // Create and configure Wi-Fi interfaces

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNodes);


  
  // Set channel width
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
               StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));

  // Install an Internet stack
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);

  TrafficControlHelper tch;
  tch.SetRootQueueDisc("ns3::FifoQueueDisc", "MaxSize",  StringValue(std::to_string(maxQueueSize)+"p"));
  tch.Install(apDevice);
  tch.Install(staDevice);
  

  // Configure IP addressing
  Ipv4AddressHelper address ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer apNodeInterface = address.Assign (apDevice);
  Ipv4InterfaceContainer staNodeInterface = address.Assign (staDevice);
  

  // PopulateArpCache
  PopulateARPcache ();

  // Configure applications
  DataRate applicationDataRate = DataRate (dataRate * 1e6);
  uint32_t portNumber = 9;

  // ap callback
  // Config::Connect ("/NodeList/"+ std::to_string(0) +"/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx", MakeCallback(&IpCheck));
  
  // callbacks conf
  for (uint32_t j = 0; j < wifiStaNodes.GetN (); ++j)
    {
      InstallTrafficGenerator (wifiStaNodes.Get (j), wifiApNode.Get (0), portNumber++,
                               applicationDataRate, packetSize);
      global_drop_list[j] = 0;
      previous_global_drop_list[j] = 0;
      Config::Connect ("/NodeList/"+ std::to_string(j+1) +"/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx", MakeCallback(&MonitorRetransmissions));
    }

  // Install FlowMonitor
  FlowMonitorHelper flowmon;
  monitor = flowmon.InstallAll ();
  csvLogOutput << "agent,dataRate,distance,nWifi,nWifiReal,seed,warmupEnd,fairness,latency,plr,throughput,time" << std::endl;

  // Generate PCAP at AP
  if (!pcapName.empty ())
    {
      phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
      phy.EnablePcap (pcapName, apDevice);
    }

  // Setup interaction with the agent
  if (agentName == "wifi")
    {
      SetNetworkConfiguration (cw_idx);
    }

  m_env->SetCond (2, 0);
  Simulator::Schedule (Seconds (fuzzTime), &ResetMonitor);
  Simulator::Schedule (Seconds (fuzzTime), &ExecuteAction, agentName, dataRate, distance, nWifi, cheaterNumber);

  // Record start time
  std::cout << "Starting simulation..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now ();
  
  // Run the simulation!
  Simulator::Run ();

  // Record stop time and count duration
  auto finish = std::chrono::high_resolution_clock::now ();
  std::chrono::duration<double> elapsed = finish - start;
  

  std::cout << "Done!" << std::endl
            << "Elapsed time: " << elapsed.count () << " s" << std::endl
            << std::endl;

  // Calculate per-flow throughput and Jain's fairness index
  double nWifiReal = 0;
  double jainsIndexN = 0.;
  double jainsIndexD = 0.;

  Time latencySum = Seconds(0);
  double lostSum = 0.;
  double txSum = 0.;
  double rxSum = 0.;

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
  std::cout << "Results: " << std::endl;

  for (auto &stat : stats)
  {
    double flow = 8 * stat.second.rxBytes / (1e6 * simulationTime);

    if (flow > 0)
      {
        nWifiReal += 1;
      }

    jainsIndexN += flow;
    jainsIndexD += flow * flow;

    latencySum += stat.second.delaySum;
    lostSum += stat.second.lostPackets;
    txSum += stat.second.txPackets;
    rxSum += stat.second.rxPackets;
    Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (stat.first);
    std::cout << "Flow " << stat.first << " (" << t.sourceAddress << " -> "
              << t.destinationAddress << ")\tThroughput: " << flow << " Mb/s" << std::endl;
  }

  double totalThr = jainsIndexN;
  double fairnessIndex = jainsIndexN * jainsIndexN / (nWifiReal * jainsIndexD);
  double totalPLR = lostSum / txSum;
  double totalLatency = latencySum.GetSeconds();
  double latencyPerPacketTotal = totalLatency / txSum;
  double normalTHR = 0;
  double cheaterTHR = 0;
  double normalAvgTHR = 0;
  double cheaterAvgTHR = 0;

  if (agentName != "wifi") {
    for (int i=1; i <= cheaterNumber; i++) {
      cheaterTHR += 8 * ( stats[i].rxBytes) / (1e6 * simulationTime);
    }
    for (uint32_t i=cheaterNumber+1; i <= nWifi; i++) {
      normalTHR += 8 * ( stats[i].rxBytes) / (1e6 * simulationTime);
    }
  
    normalAvgTHR = normalTHR / (nWifi - cheaterNumber);
    cheaterAvgTHR = cheaterTHR / (cheaterNumber);

    std::cout << std::endl
            << "Cheater throughput: " << cheaterTHR << " Mb/s" << std::endl
            << "Normal STA avg throughput: " << normalAvgTHR << " Mb/s" << std::endl;
  } else if (agentName == "wifi") {
    normalAvgTHR = totalThr / (nWifi);
    std::cout << std::endl
            << "Network avg throughput: " << normalAvgTHR << " Mb/s" << std::endl;
  }
  
  // Print results
  std::cout << std::endl
            << "Network throughput: " << totalThr << " Mb/s" << std::endl
            << "Jain's fairness index: " << fairnessIndex << std::endl
            << "PLR: " << totalPLR << std::endl
            << "Total Latency: " << totalLatency << std::endl
            << "Latency per packet: " << latencyPerPacketTotal << std::endl
            << "Total Lost packets: " << lostSum << std::endl
            << "Total tx packets: " << txSum << std::endl
            << "Total rx packets: " << rxSum << std::endl
            << std::endl;

  // Gather results in CSV format
  std::ostringstream csvOutput;
  csvOutput << "agent,dataRate,distance,nWifi,nWifiReal,seed,warmupEnd,fairness,latency,plr,throughput,cheaterTHR,cheaterAvgTHR,normalTHR,normalAvgTHR,cheaterNumber"<< std::endl;
  csvOutput << agentName << "," << dataRate << "," << distance << "," << nWifi << "," << nWifiReal << ","
            << RngSeedManager::GetRun () << "," << warmupEndTime << "," << fairnessIndex << ","
            << latencyPerPacketTotal << "," << totalPLR << "," << totalThr << "," 
            << cheaterTHR << "," << cheaterAvgTHR << "," << normalTHR << "," << normalAvgTHR << "," << cheaterNumber << std::endl;

  // Print results to files
  std::ofstream outputFile (csvPath);
  outputFile << csvOutput.str ();
  std::cout << std::endl << "Simulation data saved to: " << csvPath;

  std::ofstream outputLogFile (csvLogPath);
  outputLogFile << csvLogOutput.str ();
  std::cout << std::endl << "Simulation log saved to: " << csvLogPath << std::endl << std::endl;

  monitor->SerializeToXmlFile (flowmonPath, true, true);
  std::cout << "Flow monitor data saved to: " << flowmonPath << std::endl;


  for (uint32_t i=0; i < nWifi; i++) {
      std::cout << "Collisions packet " << i << ": " << global_drop_list[i] << std::endl;
      std::cout << "RX packets " << i << ": " << stats[i+1].rxPackets << std::endl;
      std::cout << "TX packets " << i << ": " << stats[i+1].txPackets << std::endl;
      std::cout << "LOST packets " << i << ": " << stats[i+1].lostPackets << std::endl;
  }
  std::cout << "Collisions packet " << global_collinsions_ap << std::endl;

  // for (uint32_t i = 0; i < wifiStaNodes.GetN(); ++i)
  // {
  //     Ptr<NetDevice> device = staDevice.Get(i);
  //     Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice>(device);
  //     Ptr<WifiMac> wifiMac = wifiDevice->GetMac();

  //     UintegerValue cwMinValue;
  //     // Wywołanie GetAttribute z referencją do obiektu
  //     wifiMac->GetAttribute("MinCws", cwMinValue);
  //     // Pobranie wartości z obiektu UintegerValue
  //     uint32_t cwMin = cwMinValue.Get();

  //     std::cout << "CWmin: " << cwMin << std::endl;
  // }

  // Cleanup
  Simulator::Destroy ();
  m_env->SetFinish ();

  return 0;
}

/***** Function definitions *****/

void
ResetMonitor ()
{
  monitor->CheckForLostPackets ();
  monitor->ResetAllStats ();
  previousStats = monitor->GetFlowStats ();
  previousRX = 0;
  previousTX = 0;
  previousLost = 0;
  previousDelay = Seconds(0);
}

void
InstallTrafficGenerator (Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, uint32_t port,
                         DataRate offeredLoad, uint32_t packetSize)
{
  // Get sink address
  Ptr<Ipv4> ipv4 = toNode->GetObject<Ipv4> ();
  Ipv4Address addr = ipv4->GetAddress (1, 0).GetLocal ();

  // Define type of service
  uint8_t tosValue = 0x70; //AC_BE

  // Add random fuzz to app start time
  Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable> ();
  fuzz->SetAttribute ("Min", DoubleValue (0.));
  fuzz->SetAttribute ("Max", DoubleValue (fuzzTime));
  fuzz->SetStream (0);
  double applicationsStart = fuzz->GetValue ();

  // Configure source and sink
  InetSocketAddress sinkSocket (addr, port);
  PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", sinkSocket);

  OnOffHelper onOffHelper ("ns3::UdpSocketFactory", sinkSocket);
  onOffHelper.SetConstantRate (offeredLoad, packetSize);
  onOffHelper.SetAttribute("Tos", UintegerValue(tosValue));

  // Configure applications
  ApplicationContainer sinkApplications (packetSinkHelper.Install (toNode));
  ApplicationContainer sourceApplications (onOffHelper.Install (fromNode));

  sinkApplications.Start (Seconds (applicationsStart));
  sourceApplications.Start (Seconds (applicationsStart));
}

void
PopulateARPcache ()
{
  Ptr<ArpCache> arp = CreateObject<ArpCache> ();
  arp->SetAliveTimeout (Seconds (3600 * 24 * 365));

  for (auto i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
      ObjectVectorValue interfaces;
      ip->GetAttribute ("InterfaceList", interfaces);

      for (auto j = interfaces.Begin (); j != interfaces.End (); j++)
        {
          Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface> ();
          Ptr<NetDevice> device = ipIface->GetDevice ();
          Mac48Address addr = Mac48Address::ConvertFrom (device->GetAddress ());

          for (uint32_t k = 0; k < ipIface->GetNAddresses (); k++)
            {
              Ipv4Address ipAddr = ipIface->GetAddress (k).GetLocal ();
              if (ipAddr == Ipv4Address::GetLoopback ())
                {
                  continue;
                }

              ArpCache::Entry *entry = arp->Add (ipAddr);
              Ipv4Header ipv4Hdr;
              ipv4Hdr.SetDestination (ipAddr);

              Ptr<Packet> p = Create<Packet> (100);
              entry->MarkWaitReply (ArpCache::Ipv4PayloadHeaderPair (p, ipv4Hdr));
              entry->MarkAlive (addr);
            }
        }
    }

  for (auto i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
      ObjectVectorValue interfaces;
      ip->GetAttribute ("InterfaceList", interfaces);

      for (auto j = interfaces.Begin (); j != interfaces.End (); j++)
        {
          Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface> ();
          ipIface->SetAttribute ("ArpCache", PointerValue (arp));
        }
    }
}

void
ExecuteAction (std::string agentName, double dataRate, double distance, uint32_t nWifi, int cheaterNumber)
{
  monitor->CheckForLostPackets ();
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
  double nWifiReal = 0;
  double jainsIndexNTemp = 0.;
  double jainsIndexDTemp = 0.;
  double *throughput_list = new double[cheaterNumber];
  double *tx_list = new double[cheaterNumber];
  double *lost_list = new double[cheaterNumber];
  double *collisions_list = new double[cheaterNumber];

  double currentRX = 0;
  double currentTX = 0;
  double currentLost = 0;
  Time currentDelay = Seconds (0);

  for(int i = 1; i <= cheaterNumber; i++){
    throughput_list[i-1] = 8 * ( stats[i].rxBytes - previousStats[i].rxBytes) / (1e6 * interactionTime);
    lost_list[i-1] = (stats[i].lostPackets - previousStats[i].lostPackets);
    tx_list[i-1] = ( stats[i].rxBytes - previousStats[i].rxBytes);
    collisions_list[i-1] = global_drop_list[i-1] - previous_global_drop_list[i-1];
    previous_global_drop_list[i-1] = global_drop_list[i-1];
   }
  
  previousStats = stats;

  bool end_warmup = false;

  if (useMabAgent && Simulator::Now ().GetSeconds () >= fuzzTime)
    {
      auto env = m_env->EnvSetterCond ();
      env->fairness = 0;
      env->latency = 0;
      env->plr = 0;
      for(int i = 0; i < cheaterNumber; i++){
        env->lost_list[i] = lost_list[i];
        env->tx_list[i] = tx_list[i];
        env->throughput[i] = throughput_list[i];
        env->collisions[i] = collisions_list[i];
      }
      env->time = Simulator::Now ().GetSeconds () - fuzzTime;
      m_env->SetCompleted ();

      auto act = m_env->ActionGetterCond ();
      end_warmup = act->end_warmup;
      m_env->GetCompleted ();
      for(int i = 1; i <= cheaterNumber; i++){
        int cw_idx = act->cw[i-1];
        SetNetworkConfigurationCheater (cw_idx,i);
      }
    }
  else if (!useMabAgent && Simulator::Now ().GetSeconds () >= fuzzTime)
    {
      end_warmup = true;
    }

  // End warmup period, define simulation stop time, and reset stats
  if (end_warmup && !simulationPhase)
    {
      Simulator::ScheduleNow (&ResetMonitor);
      Simulator::Stop (Seconds (simulationTime));
      simulationPhase = true;
      warmupEndTime = Simulator::Now ().GetSeconds () - fuzzTime;
      std::cout << "Warmup period finished after " << warmupEndTime << " s" << std::endl;
    }

  // csvLogOutput << agentName << "," << dataRate << "," << distance << "," << nWifi << "," << nWifiReal << "," << RngSeedManager::GetRun () << "," << end_warmup << ","
  // << fairnessIndex << "," << latencyPerPacket << "," << PLR << "," << throughput << "," << Simulator::Now().GetSeconds() - fuzzTime << std::endl;

  delete throughput_list;
  delete tx_list;
  delete lost_list;
  Simulator::Schedule (Seconds(interactionTime), &ExecuteAction, agentName, dataRate, distance, nWifi, cheaterNumber);
  // for (uint32_t j = 0; j < nWifi; ++j)
  // {
  //   Config::Connect ("/NodeList/"+ std::to_string(j) +"/DeviceList/*/$ns3::WifiNetDevice/Mac/MpduResponseTimeout", MakeCallback(&TxDrop));
  // }
  
}

void
SetNetworkConfigurationCheater (int cw_idx, int cheaterNum)
{
  if (cw_idx >= 0)
    {
      // Set CW
      AttributeContainerValue<UintegerValue> cwValue { std::vector {UintegerValue (pow (2, cw_idx))} };
      Config::Set ("/NodeList/" + std::to_string(cheaterNum) + "/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MinCws", cwValue);
      // Config::Set ("/NodeList/" + std::to_string(cheaterNum) + "/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MaxCws", cwValue);
    }
}


void
SetNetworkConfiguration (int cw_idx)
{
  if (cw_idx >= 0)
    {
      // Set CW
      AttributeContainerValue<UintegerValue> cwValueMIN { std::vector {UintegerValue (32)} };
      AttributeContainerValue<UintegerValue> cwValueMAX { std::vector {UintegerValue (1024)} };
      Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MinCws", cwValueMIN);
      Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_Txop/MaxCws", cwValueMAX);
    }
}
