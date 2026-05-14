from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
import os

logger = get_logger(__name__)

def get_kafka_config():
    cfg = load_config()
    kafka_cfg = cfg.get("kafka", {})

    # 2. İkinci öncelik settings.yaml içindeki değer
    # 3. Son çare varsayılan string
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", 
        kafka_cfg.get("bootstrap_servers", "kafka:29092")
    )
    
    return {"bootstrap.servers": bootstrap_servers}

def create_topic_if_not_exists(topic_name, num_partitions=1, replication_factor=1):
    conf = get_kafka_config()
    admin_client = AdminClient(conf)
    
    try:
        metadata = admin_client.list_topics(timeout=10)
        if topic_name not in metadata.topics:
            logger.info(f"Topic oluşturuluyor: {topic_name}")
            new_topic = NewTopic(topic_name, num_partitions, replication_factor)
            admin_client.create_topics([new_topic])[topic_name].result() # Bekleyerek oluştur
            logger.info(f"{topic_name} başarıyla hazırlandı.")
    except Exception as e:
        logger.error(f"Kafka Admin hatası: {e}")

def get_producer(client_id="bf-producer"):
    conf = get_kafka_config()
    conf.update({'client.id': client_id})
    return Producer(conf)

def get_consumer(group_id):
    conf = get_kafka_config()
    conf.update({
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    })
    return Consumer(conf)

def delivery_report(err, msg):

    if err is not None:
        logger.error(f"Mesaj iletilemedi: {err}")
    else:
        logger.debug(f"İletildi: {msg.topic()} [{msg.partition()}]")
    