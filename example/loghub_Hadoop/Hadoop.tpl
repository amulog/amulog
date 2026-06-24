Created MRAppMaster for application **
Executing with tokens
Kind YARN_AM_RM_TOKEN Service Ident appAttemptId application_id id ** cluster_timestamp ** attemptId ** keyId **
Using mapred newApiCommitter.
OutputCommitter set in config null
OutputCommitter is org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter
Registering class ** for class **
Default file system hdfs ** **
Emitting job history data to the timeline server is not enabled
loaded properties from hadoop-metrics2.properties
Scheduled snapshot period at ** second s .
MRAppMaster metrics system started
Adding job token for ** to jobTokenSecretManager
Not uberizing ** because not enabled too many maps too much input
Input size for job ** ** Number of splits **
Number of reduces for job ** **
** Transitioned from NEW to INITED
MRAppMaster launching normal non-uberized multi-container job **
Using callQueue class java.util.concurrent.LinkedBlockingQueue
Starting Socket Reader ** for port **
Adding protocol org.apache.hadoop.mapreduce.v2.api.MRClientProtocolPB to the server
Instantiated MRClientService at ** **
IPC Server Responder starting
IPC Server listener on ** starting
Logging to ** org.mortbay.log via **
Http request log for http.requests.mapreduce is not defined
Added global filter safety class org.apache.hadoop.http.HttpServer2$QuotingInputFilter
Added filter AM_PROXY_FILTER class org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter to context mapreduce
Added filter AM_PROXY_FILTER class org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter to context static
adding path spec **
Jetty bound to port **
jetty-6.1.26
Extract jar file ** ** to ** **
Started ** **
Web app /mapreduce started at **
Registered webapp guice modules
JOB_CREATE **
nodeBlacklistingEnabled true
maxTaskFailuresPerNode is **
blacklistDisablePercent is **
Connecting to ResourceManager at ** **
maxContainerCapability memory ** vCores **
queue default
Upper limit on the thread pool size is **
yarn.client.max-cached-nodemanagers-proxies **
** Transitioned from INITED to SETUP
Processing the event EventType JOB_SETUP
** Transitioned from SETUP to RUNNING
Resolved ** to /default-rack
** Task Transitioned from NEW to SCHEDULED
** TaskAttempt Transitioned from NEW to UNASSIGNED
mapResourceRequest memory ** vCores **
Event Writer setup for JobId ** File hdfs ** **
reduceResourceRequest memory ** vCores **
Before Scheduling PendingReds ** ScheduledMaps ** ScheduledReds ** AssignedMaps ** AssignedReds ** CompletedMaps ** CompletedReds ** ContAlloc ** ContRel ** HostLocal ** RackLocal **
getResources for ** ask ** release ** newContainers ** finishedContainers ** resourcelimit memory ** vCores ** knownNMs **
Recalculating schedule headroom memory ** vCores **
Reduce slow start threshold not met. completedMapsForReduceSlowstart **
Got allocated containers **
Assigned container ** to **
After Scheduling PendingReds ** ScheduledMaps ** ScheduledReds ** AssignedMaps ** AssignedReds ** CompletedMaps ** CompletedReds 0 ContAlloc ** ContRel ** HostLocal ** RackLocal **
The job-jar file on the remote FS is hdfs ** **
The job-conf file on the remote FS is **
Adding ** tokens and ** secret keys for NM use for launching container
Size of containertokens_dob is **
Putting shuffle token in serviceData
** TaskAttempt Transitioned from UNASSIGNED to ASSIGNED
Processing the event EventType ** for container ** taskAttempt **
Launching **
Opening proxy ** **
Shuffle port returned by ContainerManager for ** **
TaskAttempt ** using containerId ** on NM ** **
** TaskAttempt Transitioned from ASSIGNED to RUNNING
ATTEMPT_START **
** Task Transitioned from SCHEDULED to RUNNING
Auth successful for ** auth SIMPLE
JVM with ID ** asked for a task
JVM with ID ** given task **
Progress of TaskAttempt ** is **
Cannot assign container Container ContainerId ** NodeId ** ** NodeHttpAddress ** ** Resource memory ** vCores ** Priority ** Token Token kind ContainerToken service ** ** for a map as either container memory less than required memory ** vCores ** or no pending map tasks - maps.isEmpty true
Received completed container **
Container complete event for unknown container id **
Done acknowledgement from **
** TaskAttempt Transitioned from RUNNING to SUCCESS_CONTAINER_CLEANUP
KILLING **
** TaskAttempt Transitioned from SUCCESS_CONTAINER_CLEANUP to SUCCEEDED
Task succeeded with attempt **
** Task Transitioned from RUNNING to SUCCEEDED
Num completed Tasks **
Reduce slow start threshold reached. Scheduling reduces.
All maps assigned. Ramping up all remaining reduces **
DefaultSpeculator.addSpeculativeAttempt -- we are speculating **
We launched ** speculations. Sleeping ** milliseconds.
Scheduling a redundant attempt for task **
Diagnostics report from ** Container killed by the ApplicationMaster.
Address change detected. Old ** ** New ** **
Failed to renew lease for ** for ** seconds. Will retry shortly ...
Slow ReadProcessor read fields took ** threshold ** ack seqno ** status SUCCESS status ERROR downstreamAckTimeNanos ** targets ** ** ** **
DFSOutputStream ResponseProcessor exception for block ** **
Error Recovery for block ** ** in pipeline ** ** ** ** bad datanode ** **
DataStreamer Exception
ERROR IN CONTACTING RM.
Retrying connect to server ** ** Already tried ** time s retry policy is RetryUpToMaximumCountWithFixedSleep maxRetries ** sleepTime ** MILLISECONDS
Task ** - exited java.net.NoRouteToHostException No Route to Host from ** to ** ** failed on socket timeout exception java.net.NoRouteToHostException No route to host no further information For more details see http //wiki.apache.org/hadoop/NoRouteToHost
Diagnostics report from ** Error java.net.NoRouteToHostException No Route to Host from ** to ** ** failed on socket timeout exception java.net.NoRouteToHostException No route to host no further information For more details see http //wiki.apache.org/hadoop/NoRouteToHost
** TaskAttempt Transitioned from RUNNING to FAIL_CONTAINER_CLEANUP
** TaskAttempt Transitioned from FAIL_CONTAINER_CLEANUP to FAIL_TASK_CLEANUP
Processing the event EventType TASK_ABORT
Task cleanup failed for attempt **
** TaskAttempt Transitioned from FAIL_TASK_CLEANUP to FAILED
Error writing History Event **
Thread Thread eventHandlingThread ** main threw an Exception.
** failures on node **
Added ** to list of failed maps
