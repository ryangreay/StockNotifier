import stripe
from datetime import datetime
from fastapi import HTTPException, Request
import os
from sqlalchemy.orm import Session
from . import models

# Initialize Stripe with your secret key
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

async def handle_subscription_created(event: dict, db: Session):
    """Handle subscription.created event"""
    subscription = event['data']['object']
    customer_id = subscription['customer']
    
    # Find user by Stripe customer ID
    user = db.query(models.User).filter(
        models.User.stripe_customer_id == customer_id
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user's subscription information
    user.stripe_subscription_id = subscription['id']
    user.subscription_status = subscription['status']
    user.subscription_plan_id = subscription['plan']['id']
    user.subscription_end_date = datetime.fromtimestamp(subscription['current_period_end'])
    
    db.commit()

async def handle_subscription_updated(event: dict, db: Session):
    """Handle subscription.updated event"""
    subscription = event['data']['object']
    
    # Find user by subscription ID
    user = db.query(models.User).filter(
        models.User.stripe_subscription_id == subscription['id']
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update subscription information
    user.subscription_status = subscription['status']
    user.subscription_end_date = datetime.fromtimestamp(subscription['current_period_end'])
    
    # If plan changed, update plan ID
    if subscription.get('plan'):
        user.subscription_plan_id = subscription['plan']['id']
    
    db.commit()

async def handle_subscription_deleted(event: dict, db: Session):
    """Handle subscription.deleted event"""
    subscription = event['data']['object']
    
    # Find user by subscription ID
    user = db.query(models.User).filter(
        models.User.stripe_subscription_id == subscription['id']
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update subscription information
    user.subscription_status = 'canceled'
    user.subscription_end_date = datetime.fromtimestamp(subscription['current_period_end'])
    
    db.commit()

async def handle_customer_subscription_trial_will_end(event: dict, db: Session):
    """Handle customer.subscription.trial_will_end event"""
    subscription = event['data']['object']
    
    # Find user by subscription ID
    user = db.query(models.User).filter(
        models.User.stripe_subscription_id == subscription['id']
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # You could send an email notification here or update some notification flag
    # For now, we'll just update the subscription end date
    user.subscription_end_date = datetime.fromtimestamp(subscription['trial_end'])
    
    db.commit()

async def handle_payment_failed(event: dict, db: Session):
    """Handle invoice.payment_failed event"""
    invoice = event['data']['object']
    customer_id = invoice['customer']
    
    # Find user by customer ID
    user = db.query(models.User).filter(
        models.User.stripe_customer_id == customer_id
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update subscription status to reflect payment failure
    user.subscription_status = 'past_due'
    
    db.commit()

# Map Stripe events to their handlers
event_handlers = {
    'customer.subscription.created': handle_subscription_created,
    'customer.subscription.updated': handle_subscription_updated,
    'customer.subscription.deleted': handle_subscription_deleted,
    'customer.subscription.trial_will_end': handle_customer_subscription_trial_will_end,
    'invoice.payment_failed': handle_payment_failed,
}

async def handle_stripe_webhook(request: Request, db: Session):
    """Main webhook handler"""
    # Get the raw payload and signature header
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")
    
    try:
        # Verify webhook signature and parse the event
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Get the handler for this event type
    event_handler = event_handlers.get(event['type'])
    if event_handler:
        try:
            await event_handler(event, db)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Ignore events we're not handling
        return {"status": "ignored"} 